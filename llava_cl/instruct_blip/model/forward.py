import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Union

import transformers
from transformers.models.instructblip.modeling_instructblip import InstructBlipForConditionalGenerationModelOutput


def forward(
    self,
    pixel_values: torch.FloatTensor,
    qformer_input_ids: torch.FloatTensor,
    qformer_attention_mask: Optional[torch.LongTensor] = None,
    input_ids: Optional[torch.FloatTensor] = None,
    attention_mask: Optional[torch.LongTensor] = None,
    decoder_input_ids: Optional[torch.LongTensor] = None,
    decoder_attention_mask: Optional[torch.LongTensor] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    labels: Optional[torch.LongTensor] = None,
    prompts: Optional[List[str]] = None,
    answers: Optional[List[str]] = None,
    output_labels: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, InstructBlipForConditionalGenerationModelOutput]:
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # step 1: forward the images through the vision encoder,
    # to get image embeddings of shape (batch_size, seq_len, hidden_size)
    vision_outputs = self.vision_model(
        pixel_values=pixel_values,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )
    image_embeds = vision_outputs[0]

    # step 2: forward the query tokens through the QFormer, using the image embeddings for cross-attention
    image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

    # difference with BLIP-2 here: we also feed the instruction prompt to the Q-Former
    query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
    query_attention_mask = torch.ones(query_tokens.size()[:-1], dtype=torch.long, device=image_embeds.device)
    if qformer_attention_mask is None:
        qformer_attention_mask = torch.ones_like(qformer_input_ids)
    qformer_attention_mask = torch.cat([query_attention_mask, qformer_attention_mask], dim=1)
    query_outputs = self.qformer(
        input_ids=qformer_input_ids,
        attention_mask=qformer_attention_mask,
        query_embeds=query_tokens,
        encoder_hidden_states=image_embeds,
        encoder_attention_mask=image_attention_mask,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )
    query_output = query_outputs[0][:, : query_tokens.size(1), :]

    # step 3: use the language model, conditioned on the query outputs and the prompt
    if hasattr(self, "e_language_projection"):
        language_model_inputs = self.e_language_projection(query_output, image_embeds[:, 0, :], prompts)
    else:
        language_model_inputs = self.language_projection(query_output)
    language_model_attention_mask = torch.ones(
        language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
    )

    inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

    inputs_embeds = torch.cat([language_model_inputs, inputs_embeds.to(language_model_inputs.device)], dim=1)

    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    attention_mask = torch.cat([language_model_attention_mask.to(attention_mask.device), attention_mask], dim=1)

    if self.config.use_decoder_only_language_model:
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits if return_dict else outputs[0]
        loss = None
        # we compute the loss here since we need to take into account the sequence length of the query embeds
        if labels is not None:
            labels = labels.to(logits.device)
            logits = logits[:, -labels.size(1) :, :]
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous().to(logits.device)

            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss(reduction="mean")

            loss = loss_fct(shift_logits.view(-1, self.config.text_config.vocab_size), shift_labels.view(-1))
    else:
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            labels=labels,
        )
        loss = outputs.loss if return_dict else outputs[0]
        logits = outputs.logits if return_dict else outputs[1]

    if not return_dict:
        output = (logits, vision_outputs, query_outputs, outputs)
        return ((loss,) + output) if loss is not None else output

    outputs = InstructBlipForConditionalGenerationModelOutput(
        loss=loss,
        logits=logits,
        vision_outputs=vision_outputs,
        qformer_outputs=query_outputs,
        language_model_outputs=outputs,
    )

    if output_labels:
        return outputs, labels
    else:
        return outputs


@torch.no_grad()
def generate(
    self,
    pixel_values: torch.FloatTensor,
    qformer_input_ids: Optional[torch.LongTensor] = None,
    qformer_attention_mask: Optional[torch.LongTensor] = None,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.LongTensor] = None,
    prompts: Optional[List[str]] = None,
    answers: Optional[List[str]] = None,
    **generate_kwargs,
) -> torch.LongTensor:

    if hasattr(self, "hf_device_map"):
        # preprocess for `accelerate`
        self._preprocess_accelerate()

    batch_size = pixel_values.shape[0]
    image_embeds = self.vision_model(pixel_values, return_dict=True).last_hidden_state

    image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

    query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
    query_attention_mask = torch.ones(query_tokens.size()[:-1], dtype=torch.long, device=image_embeds.device)
    if qformer_attention_mask is None:
        qformer_attention_mask = torch.ones_like(qformer_input_ids)
    qformer_attention_mask = torch.cat([query_attention_mask, qformer_attention_mask], dim=1)
    query_outputs = self.qformer(
        input_ids=qformer_input_ids,
        attention_mask=qformer_attention_mask,
        query_embeds=query_tokens,
        encoder_hidden_states=image_embeds,
        encoder_attention_mask=image_attention_mask,
        return_dict=True,
    )
    query_output = query_outputs.last_hidden_state[:, : query_tokens.size(1), :]

    if hasattr(self, "e_language_projection"):
        language_model_inputs = self.e_language_projection(query_output, image_embeds[:, 0, :], prompts)
    else:
        language_model_inputs = self.language_projection(query_output)

    language_attention_mask = torch.ones(
        language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
    )

    if input_ids is None:
        input_ids = (
            torch.LongTensor([[self.config.text_config.bos_token_id]])
            .repeat(batch_size, 1)
            .to(image_embeds.device)
        )
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    attention_mask = torch.cat([language_attention_mask, attention_mask.to(language_attention_mask.device)], dim=1)

    # concatenate query embeddings with prompt embeddings
    inputs_embeds = self.get_input_embeddings()(input_ids)
    inputs_embeds = torch.cat([language_model_inputs, inputs_embeds.to(language_model_inputs.device)], dim=1)

    outputs = self.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        **generate_kwargs,
    )

    # the InstructBLIP authors used inconsistent tokenizer/model files during training,
    # with the tokenizer's bos token being set to </s> which has ID=2,
    # whereas the model's text config has bos token id = 0
    if self.config.text_config.architectures[0] == "LLaMAForCausalLM":
        if isinstance(outputs, torch.Tensor):
            outputs[outputs == 0] = 2
        else:
            outputs.sequences[outputs.sequences == 0] = 2

    return outputs


def replace_instruct_blip_forward():
    transformers.models.instructblip.modeling_instructblip.InstructBlipForConditionalGeneration.forward = forward

def replace_instruct_blip_generate():
    transformers.models.instructblip.modeling_instructblip.InstructBlipForConditionalGeneration.generate = generate

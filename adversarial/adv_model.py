import torch
from torch.nn import CrossEntropyLoss
from typing import Optional, Tuple, Union
from dataclasses import dataclass
from .adv_head import AdversarialClassifierHead
from .reg_args import RegArguments
from transformers.modeling_outputs import ModelOutput
from transformers import BertForSequenceClassification, BertForMaskedLM
from transformers.adapters import BertAdapterModel

@dataclass
class AdvMaskedLMOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    adv_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class AdvSequenceClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    adv_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class AdvBertForSequenceClassification(BertForSequenceClassification):
    def __init__(self, config, reg_args, cls_weights=None):
        super().__init__(config)
        # adversarial training variables
        self.adv_debias = False
        self.reg_args = None
        self.adv_model = None
        self.finetune = False
        self.cls_weights = None
        if cls_weights is not None:
            self.cls_weights = torch.tensor(cls_weights).float()
        self.build_adv_training(reg_args)
    
    def build_adv_training(self, reg_args: RegArguments):
        assert self.adv_model is None
        self.reg_args = reg_args
        if reg_args.adv_debias:
            self.adv_debias = True
            self.adv_model = AdversarialClassifierHead(
                                self.config.hidden_size, 
                                attr_dim=reg_args.adv_attr_dim, 
                                adv_dropout=reg_args.adv_dropout,
                                hidden_layer_num=reg_args.adv_layer_num,
                                adv_grad_rev_strength=reg_args.adv_grad_rev_strength
                            )
        if reg_args.finetune:
            self.finetune = True

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        return_dict=None,
        labels=None,
        attr=None
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            return_dict=return_dict
        )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        adv_loss, attr_logits, attr_probs = None, None, None

        loss = 0.0

        if labels is not None:
            if self.finetune:
                if self.num_labels == 1:
                    #  We are doing regression
                    loss_fct = MSELoss()
                    loss = loss_fct(logits.view(-1), labels.view(-1))
                else:
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            if self.adv_debias:
                reg_args = self.reg_args
                attr_logits = self.adv_model(pooled_output, rev_grad=True)
                # attr_probs = torch.nn.Softmax(-1)(attr_logits)
                if self.cls_weights is not None:
                    adv_loss = self.adv_model.compute_loss(attr_pred=attr_logits, attr_gt=attr, cls_weights=self.cls_weights.to(self.bert.device))
                else:
                    adv_loss = self.adv_model.compute_loss(attr_pred=attr_logits, attr_gt=attr)
                loss += adv_loss * reg_args.adv_strength

        if not return_dict:
            output = outputs[2:]
            if self.finteune:
                output = (logits,) + output
            if self.adv_debias:
                output = (attr_logits,) + output
            return ((loss,) + output) if loss is not None else output
        
        return AdvSequenceClassifierOutput(
            loss=loss,
            logits=attr_logits
        )


class AdvBertForMaskedLM(BertForMaskedLM):

    def __init__(self, config, reg_args):
        super().__init__(config)
        self.adv_debias = False
        self.adv_model = None
        self.build_adv_training(reg_args)

    def build_adv_training(self, reg_args: RegArguments):
        assert self.adv_model is None
        self.reg_args = reg_args
        if reg_args.adv_debias:
            self.adv_debias = True
            self.adv_model = AdversarialClassifierHead(
                                self.config.hidden_size, 
                                attr_dim=reg_args.adv_attr_dim, 
                                adv_dropout=reg_args.adv_dropout,
                                hidden_layer_num=reg_args.adv_layer_num,
                                adv_grad_rev_strength=reg_args.adv_grad_rev_strength
                            )

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        attr: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor], AdvMaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(
            sequence_output,
            inv_lang_adapter=self.bert.get_invertible_adapter()
        )

        masked_lm_loss = None
        adv_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            
            if self.adv_debias and attr is not None:
                reg_args = self.reg_args
                attr_logits = self.adv_model(sequence_output[:,0,:], rev_grad=True)
                attr_probs = torch.nn.Softmax(-1)(attr_logits)
                adv_loss = self.adv_model.compute_loss(attr_pred=attr_logits, attr_gt=attr)
                masked_lm_loss += adv_loss * reg_args.adv_strength

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
            
        return AdvMaskedLMOutput(
            loss=masked_lm_loss,
            adv_loss=adv_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

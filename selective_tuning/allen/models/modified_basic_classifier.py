from typing import Dict, Optional

from overrides import overrides
import torch
import backpack

from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TextFieldEmbedder
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder
from allennlp.modules.seq2vec_encoders import BertPooler
from allennlp.nn import InitializerApplicator
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.common.checks import ConfigurationError

def add_backpack(a_module) -> None:
    if hasattr(a_module, '_modules'):
        all_sub_modules = a_module._modules
        if len(all_sub_modules) == 0:
            print(a_module)
            return
        for module in all_sub_modules.values():
            if isinstance(module, torch.nn.modules.linear.Linear):
                print(f'{module}     YES')
                backpack.extend(module)
            #elif isinstance(module, torch.nn.Embedding):
            #    print(f'{module}     YES')
            #    backpack.extend(module)
            #elif isinstance(module, torch.nn.LayerNorm):
            #    print(f'{module}     YES')
            #    backpack.extend(module)
            else:
                add_backpack(module)
    else:
        return 
            

@Model.register("modified_basic_classifier")
class ModifiedBasicClassifier(Model):
    """
    This `Model` implements a basic text classifier. After embedding the text into
    a text field, we will optionally encode the embeddings with a `Seq2SeqEncoder`. The
    resulting sequence is pooled using a `Seq2VecEncoder` and then passed to
    a linear classification layer, which projects into the label space. If a
    `Seq2SeqEncoder` is not provided, we will pass the embedded text directly to the
    `Seq2VecEncoder`.
    # Parameters
    vocab : `Vocabulary`
    text_field_embedder : `TextFieldEmbedder`
        Used to embed the input text into a `TextField`
    dropout : `float`, optional (default = `None`)
        Dropout percentage to use.
    num_labels : `int`, optional (default = `None`)
        Number of labels to project to in classification layer. By default, the classification layer will
        project to the size of the vocabulary namespace corresponding to labels.
    label_namespace : `str`, optional (default = "labels")
        Vocabulary namespace corresponding to labels. By default, we use the "labels" namespace.
    initializer : `InitializerApplicator`, optional (default=`InitializerApplicator()`)
        If provided, will be used to initialize the model parameters.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: Optional[TextFieldEmbedder] = None,
        dropout: float = None,
        num_labels: int = None,
        label_namespace: str = "labels",
        initializer: InitializerApplicator = InitializerApplicator(),
        **kwargs,
    ) -> None:

        super().__init__(vocab, **kwargs)

        if dropout:
            self._dropout = torch.nn.Dropout(dropout)
        else:
            self._dropout = None
        self._label_namespace = label_namespace

        if num_labels:
            self._num_labels = num_labels
        else:
            self._num_labels = vocab.get_vocab_size(namespace=self._label_namespace)
        
        self._text_field_embedder = text_field_embedder
        #add_backpack(self._text_field_embedder)
        backpack.extend(list(self._text_field_embedder.modules())[-6])
        # -6
        
        self.context_encoder = BagOfEmbeddingsEncoder(text_field_embedder.get_output_dim(), 
                                                      averaged=True)
        self.other_layer = backpack.extend(torch.nn.Linear(self.context_encoder.get_output_dim(), 
                                                                     300))
        self.norming = backpack.extend(torch.nn.LayerNorm(300))
        self.a_func = torch.nn.Tanh()
        self._classification_layer = backpack.extend(torch.nn.Linear(300, 
                                                                     self._num_labels))
        self._accuracy = CategoricalAccuracy()
        self._loss = backpack.extend(torch.nn.CrossEntropyLoss())
        initializer(self)

    def forward(  # type: ignore
        self, tokens: TextFieldTensors, label: torch.IntTensor = None
    ) -> Dict[str, torch.Tensor]:

        """
        # Parameters
        tokens : TextFieldTensors
            From a `TextField`
        label : torch.IntTensor, optional (default = None)
            From a `LabelField`
        # Returns
        An output dictionary consisting of:
        logits : torch.FloatTensor
            A tensor of shape `(batch_size, num_labels)` representing
            unnormalized log probabilities of the label.
        probs : torch.FloatTensor
            A tensor of shape `(batch_size, num_labels)` representing
            probabilities of the label.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
       
        embedded_text = self._text_field_embedder(tokens)
        mask = get_text_field_mask(tokens).float()

        embedded_text = self.context_encoder(embedded_text, mask=mask)

        if self._dropout:
            embedded_text = self._dropout(embedded_text)
        c = self.a_func(self.norming(self.other_layer(embedded_text)))
        logits = self._classification_layer(c)
        probs = torch.nn.functional.softmax(logits, dim=-1)

        output_dict = {"logits": logits, "probs": probs}

        if label is not None:
            loss = self._loss(logits, label.long().view(-1))
            output_dict["loss"] = loss
            self._accuracy(logits, label)

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does a simple argmax over the probabilities, converts index to string label, and
        add `"label"` key to the dictionary with the result.
        """
        predictions = output_dict["probs"]
        if predictions.dim() == 2:
            predictions_list = [predictions[i] for i in range(predictions.shape[0])]
        else:
            predictions_list = [predictions]
        classes = []
        for prediction in predictions_list:
            label_idx = prediction.argmax(dim=-1).item()
            label_str = self.vocab.get_index_to_token_vocabulary(self._label_namespace).get(
                label_idx, str(label_idx)
            )
            classes.append(label_str)
        output_dict["label"] = classes
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {"accuracy": self._accuracy.get_metric(reset)}
        return metrics
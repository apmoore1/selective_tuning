{
  "dataset_reader":{
    "type": "sst_tokens",
    "use_subtrees": false,
    "granularity": "5-class",
    "token_indexers": {
        "tokens": {
          "type": "pretrained_transformer_mismatched",
          "model_name": "bert-base-cased"
        }
    }
  },
  "train_data_path": "./data/sst/trees/train.txt",
  "validation_data_path": "./data/sst/trees/dev.txt",
  "test_data_path": "./data/sst/trees/test.txt",  
  "evaluate_on_test": true,
  "model": {
    "type": "basic_classifier",        
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "pretrained_transformer_mismatched",
          "model_name": "bert-base-cased"
          }
      }
    },
    "seq2vec_encoder": {"type":"boe", "embedding_dim":768, "averaged": true}
  },    
  "iterator": {
    "type": "basic",
    "batch_size" : 1
  },
  "trainer": {
    "type": "modified_default",
    "num_epochs": 5,
    "patience": 1,
    "validation_metric": "+accuracy",
    "cuda_device": 0,
    "num_gradient_accumulation_steps": 4,
    "optimizer": {
      "type": "adam",
      "lr": 0.00002,
      "parameter_groups": [
        [[".*transformer_model.embeddings.*"], {}],
        [[".*transformer_model.encoder.layer.0[.].*"], {}],
        [[".*transformer_model.encoder.layer.1[.].*"], {}],
        [[".*transformer_model.encoder.layer.2[.].*"], {}],
        [[".*transformer_model.encoder.layer.3[.].*"], {}],
        [[".*transformer_model.encoder.layer.4[.].*"], {}],
        [[".*transformer_model.encoder.layer.5[.].*"], {}],
        [[".*transformer_model.encoder.layer.6[.].*"], {}],
        [[".*transformer_model.encoder.layer.7[.].*"], {}],
        [[".*transformer_model.encoder.layer.8[.].*"], {}],
        [[".*transformer_model.encoder.layer.9[.].*"], {}],
        [[".*transformer_model.encoder.layer.10[.].*"], {}],
        [[".*transformer_model.encoder.layer.11[.].*"], {}],
        [["_seq2vec_encoder.*", "_classification_layer.*", 
          ".*transformer_model.pooler.*"], {}]]
    },
    "checkpointer": {"num_serialized_models_to_keep":1}
  }
}
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
    "batch_size" : 16
  },
  "trainer": {
    "num_epochs": 5,
    "patience": 1,
    "grad_norm": 5.0,
    "validation_metric": "+accuracy",
    "cuda_device": 0,
    "optimizer": {
      "type": "adam",
      "lr": 0.00002
    },
    "checkpointer": {"num_serialized_models_to_keep":1}
  }
}
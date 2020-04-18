{
  "dataset_reader":{
    "type": "sst_tokens",
    "use_subtrees": false,
    "granularity": "5-class",
    "token_indexers": {
        "tokens": {
          "type": "single_id",
          "lowercase_tokens": true
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
          "type": "embedding",
          "embedding_dim": 50,          
          "trainable": true
          }
      }
    },
    "seq2vec_encoder": {"type":"boe", "embedding_dim":50, "averaged": true}
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
    "num_gradient_accumulation_steps": 16,
    "optimizer": {
      "type": "adam",
      "lr": 0.001
    },
    "checkpointer": {"num_serialized_models_to_keep":1}
  }
}
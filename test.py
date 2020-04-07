from pathlib import Path

from allennlp.data.dataset_readers.stanford_sentiment_tree_bank import StanfordSentimentTreeBankDatasetReader
from allennlp.data.token_indexers.single_id_token_indexer import SingleIdTokenIndexer
indexer = SingleIdTokenIndexer()
reader = StanfordSentimentTreeBankDatasetReader({'tokens': indexer})
train_data = reader.read(Path('./data/sst/trees/train.txt'))
dev_data = reader.read(Path('./data/sst/trees/dev.txt'))
test_data = reader.read(Path('./data/sst/trees/test.txt'))
print(f'Train: {len(train_data)}')
print(f'Dev: {len(dev_data)}')
print(f'Test: {len(test_data)}')
print('done')
#dev_data
#test_data =
from lit_nlp import lit_types
from lit_nlp import lit_types

class MultiNLIData(Dataset):
  """Loader for MultiNLI development set."""

  NLI_LABELS = ['entailment', 'neutral', 'contradiction']

  def __init__(self, path):
    # Read the eval set from a .tsv file as distributed with the GLUE benchmark.
    df = pandas.read_csv(path, sep='\t')
    # Store as a list of dicts, conforming to self.spec()
    self._examples = [{
      'premise': row['sentence1'],
      'hypothesis': row['sentence2'],
      'label': row['gold_label'],
      'genre': row['genre'],
    } for _, row in df.iterrows()]

  def spec(self):
    return {
      'premise': lit_types.TextSegment(),
      'hypothesis': lit_types.TextSegment(),
      'label': lit_types.Label(vocab=self.NLI_LABELS),
      # We can include additional fields, which don't have to be used by the model.
      'genre': lit_types.Label(),
    }

def main(_):
  # MulitiNLIData implements the Dataset API
  datasets = {
      'mnli_matched': MultiNLIData('/path/to/dev_matched.tsv'),
      'mnli_mismatched': MultiNLIData('/path/to/dev_mismatched.tsv'),
  }

  # NLIModel implements the Model API
  models = {
      'model_foo': NLIModel('/path/to/model/foo/files'),
      'model_bar': NLIModel('/path/to/model/bar/files'),
  }

  lit_demo = lit_nlp.dev_server.Server(models, datasets, port=4321)
  lit_demo.serve()

if __name__ == '__main__':
  main()



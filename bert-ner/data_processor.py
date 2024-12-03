class Data_Processor:

    def __init__(self, data) -> None:
        self.data = data

    def process(self):

        label2id = {}
        id2label = {}

        self.data = self.data.ffill()
        self.data['base_tag'] = self.data['Tag'].apply(lambda x: x.split('-')[-1])

        remove_tags = ['art','nat','eve']
        self.data = self.data[~self.data.base_tag.isin(remove_tags)]

        labels = self.data['Tag'].value_counts().index
        for idx, label in enumerate(labels):
            label2id[label] = idx
            id2label[idx] = label

        self.data['Sentence'] = self.data[['Sentence #','Word', "Tag"]].groupby(['Sentence #'])["Word"].transform(lambda x: ' '.join(x))
        self.data['Word_labels'] = self.data[['Sentence #','Word', "Tag"]].groupby(['Sentence #'])["Tag"].transform(lambda x: ' '.join(x))

        self.data = self.data[["Sentence", "Word_labels"]].drop_duplicates().reset_index(drop=True)

        return self.data, label2id, id2label




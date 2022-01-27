import pandas as pd
import pickle
import random as rd
from modules.nn_analogy_solver.solver import Solver

class Interface():
    def __init__(self):
        '''Stores the values received from the web page and produces needed outputs.
        - Stores the words A, B, C and D when it is displayed on screen, as well as the
        chosen features;
        - Stores the list of analogies and can output an example based on the options
        chosen on the web page;
        - Handles the analogy solving part.
        '''
        super(Interface).__init__()
        self.source_language = None
        self.target_language = None
        self.A = None
        self.B = None
        self.C = None
        self.D = None
        self.features = None

        self.prepare_data()
        self.solver = Solver()

    def prepare_data(self):
        '''Load the DataFrame of analogies.
        '''
        with open('modules/interface/data/all_analogies_df_mini', 'rb') as f:
            self.data = pickle.load(f)

    def get_example(self, possible_features = None):
        '''Returns an example based on the chosen options (lanugages, features).
        Arguments:
        possible_features -- the set of possible features for the returned example.
        '''
        previous_example = (self.A,self.B,self.C)

        # select based on languages
        if self.source_language is None and self.target_language is None:
            possible_rows = self.data
        elif self.source_language is None:
            possible_rows = self.data[(self.data['target_language']==self.target_language)]
        elif self.target_language is None:
            possible_rows = self.data[(self.data['source_language']==self.source_language)]
        else:
            possible_rows = self.data[(self.data['source_language']==self.source_language) & (self.data['target_language']==self.target_language)]

        if possible_features is not None:
            possible_rows = possible_rows[possible_rows.apply(lambda c: c['features'] in possible_features, axis=1)]

        # select based on the previous analogy
        if possible_rows.shape[0]:
            new_example = possible_rows.sample()
        else:
            return None
        if possible_rows.shape[0] > 1:
            while new_example['A'].values[0] in previous_example and new_example['B'].values[0] in previous_example and new_example['C'].values[0] in previous_example:
                new_example = possible_rows.sample()

        return new_example#possible_rows.sample()#[['A', 'B', 'C', 'D']]

    def check_example_exists(self, word_a, word_b, word_c, word_d):
        '''Check if the input example exists in the database.
        It should always return True in our case as users cannot enter an example
        manually.
        Arguments:
        word_a -- The first word in the analogy;
        word_b -- The second word in the analogy;
        word_c -- The third word in the analogy;
        word_d (optional) -- The fourth word in the analogy.
        '''
        if word_d is None:
            matching_rows = self.data[((self.data['A'] == word_a) &     (self.data['B'] == word_b) &    (self.data['C'] == word_c)                                  ) |
                                        ((self.data['A'] == word_b) &   (self.data['B'] == word_a) &                                    (self.data['D'] == word_c)  ) |
                                        ((self.data['A'] == word_a) &   (self.data['B'] == word_c) &    (self.data['C'] == word_b)                                  ) |
                                        ((self.data['A'] == word_c) &   (self.data['B'] == word_a) &                                    (self.data['D'] == word_b)  ) |
                                        ((self.data['A'] == word_c) &                                   (self.data['C'] == word_a) &    (self.data['D'] == word_b)  ) |
                                        ((self.data['A'] == word_b) &                                   (self.data['C'] == word_a) &    (self.data['D'] == word_c)  ) |
                                        (                               (self.data['B'] == word_b) &    (self.data['C'] == word_c) &    (self.data['D'] == word_a)  ) |
                                        (                               (self.data['B'] == word_c) &    (self.data['C'] == word_b) &    (self.data['D'] == word_a)  )]
        else:
            matching_rows = self.data[((self.data['A'] == word_a) & (self.data['B'] == word_b) & (self.data['C'] == word_c) & (self.data['D'] == word_d)) |
                                        ((self.data['A'] == word_b) & (self.data['B'] == word_a) & (self.data['C'] == word_d) & (self.data['D'] == word_c)) |
                                        ((self.data['A'] == word_a) & (self.data['B'] == word_c) & (self.data['C'] == word_b) & (self.data['D'] == word_d)) |
                                        ((self.data['A'] == word_c) & (self.data['B'] == word_a) & (self.data['C'] == word_d) & (self.data['D'] == word_b)) |
                                        ((self.data['A'] == word_c) & (self.data['B'] == word_d) & (self.data['C'] == word_a) & (self.data['D'] == word_b)) |
                                        ((self.data['A'] == word_b) & (self.data['B'] == word_d) & (self.data['C'] == word_a) & (self.data['D'] == word_c)) |
                                        ((self.data['A'] == word_d) & (self.data['B'] == word_b) & (self.data['C'] == word_c) & (self.data['D'] == word_a)) |
                                        ((self.data['A'] == word_d) & (self.data['B'] == word_c) & (self.data['C'] == word_b) & (self.data['D'] == word_a))]
        return matching_rows.shape[0] > 0

    def shuffle(self):
        '''Shuffle the stored analogy.
        Source shuffles:
            c, d, a, b
            c, a, d, b
            d, b, c, a
            d, c, b, a
        Target shuffles:
            a, b, c, d
            b, a, d, c
            a, c, b, d
            b, d, a, c
        '''
        n = rd.randint(0,6)
        if n == 0:
            self.A, self.B, self.C, self.D = self.B, self.A, self.D, self.C
        elif n == 1:
            self.A, self.B, self.C, self.D = self.A, self.C, self.B, self.D
        elif n == 2:
            self.A, self.B, self.C, self.D = self.B, self.D, self.A, self.C
        elif n == 3:
            self.A, self.B, self.C, self.D = self.C, self.D, self.A, self.B
            self.source_language, self.target_language = self.target_language, self.source_language
        elif n == 4:
            self.A, self.B, self.C, self.D = self.C, self.A, self.D, self.B
            self.source_language, self.target_language = self.target_language, self.source_language
        elif n == 5:
            self.A, self.B, self.C, self.D = self.D, self.B, self.C, self.A
            self.source_language, self.target_language = self.target_language, self.source_language
        else:
            self.A, self.B, self.C, self.D = self.D, self.C, self.B, self.A
            self.source_language, self.target_language = self.target_language, self.source_language

    def solve(self):
        '''Return the result of the analogy on screen.
        '''
        if self.A is not None and self.B is not None and self.C is not None:
            result = self.solver.solve(self.A, self.B, self.C)
            return result

    def get_features_list(self, searched_feature):
        '''Return a tuple with all the possible sets of features containing the pattern chosen by the user.
        The searched feature is splitted in order to get the different features wanted
        by the user. An example of input would be 'Verb, past tense', where the wanted
        features are 'Verb' and 'past tense'. It is not necessary to write a feature
        entirely to trigger it, 'past' would trigger all the features starting with start.
        Arguments:
        searched_feature: the pattern wanted by the user.
        '''
        if not searched_feature:
            return None
        searched_feature = searched_feature.replace(", ", ",")
        searched_feature_list = searched_feature.split(',')

        all_features = set(self.data['features'].values)

        possible_features = set()
        for feature in all_features:
            feature = ' ' + feature
            if all([f in feature for f in searched_feature_list]):
                possible_features.add(feature[1:])

        return tuple(possible_features)


    def get_possible_features(self, as_html=True):
        '''Return the set of possible features for the chosen lanugage pair. It can be
        returned directly in html format.
        Arguments:
        as_html -- If True, the output is in html format.
        '''
        if self.source_language is None and self.target_language is None:
            features = set(self.data['features'])
        elif self.source_language is None:
            features = set(self.data[(self.data['target_language']==self.target_language)]['features'])
        elif self.target_language is None:
            features = set(self.data[(self.data['source_language']==self.source_language)]['features'])
        else:
            features = set(self.data[(self.data['source_language']==self.source_language) & (self.data['target_language']==self.target_language)]['features'])

        if as_html:
            s = '<a name="{f}" href="#features_{id}" id=f"{id}" onclick="selectFunction(this.innerHTML)">{f}</a>\n'
            html_features = [s.format(id=i, f=f) for i,f in enumerate(features)]
            return ''.join(html_features), len(html_features)
        else:
            return features





if __name__ == '__main__':
    software = Interface()
    #software.features = 'Verb, indicative mood, indefinite, informal register, present tense, second person, singular'
    #LANGUAGES = ['finnish', 'german', 'hungarian', 'spanish', 'turkish', None]
    ex = software.get_example()
    software.A = ex['A'].values[0]
    software.B = ex['B'].values[0]
    software.C = ex['C'].values[0]
    print(f'{software.A}, {software.B}, {software.C} -> {software.solve()}')



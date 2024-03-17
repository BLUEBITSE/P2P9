from collections import OrderedDict
import random
import csv


class MCQGenerator:
    def __init__(self):
        self.nltk = None
        self.stoplist = None
        self.requests = None
        self.cache = OrderedDict()  # Fixed-size cache with LRU eviction policy
        self.cache_size_limit = 1000  # Define cache size limit


    def lazy_import_nltk(self):
        if not self.nltk:
            import nltk
            self.nltk = nltk
            self.nltk.download('stopwords')
            self.nltk.download('punkt')
            self.nltk.download('wordnet')
            self.nltk.download('averaged_perceptron_tagger')
            if self.nltk.data.find('corpora/wordnet.zip'):
                print("WordNet is installed.")
            else:
                print("WordNet is not installed. You may need to run nltk.download('wordnet').")
                self.nltk.download('wordnet')

    def get_stoplist(self):
        if not hasattr(self, 'stop_words'):
            self.lazy_import_nltk()

            import string
            stop_words = set(self.nltk.corpus.stopwords.words())
            punctuation = set(string.punctuation)
            additional_words = {'-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-'} | {
                'example', 'examples', 'task', 'entity', 'data', 'use', 'type', 'concepts', 'concept',
                'learn', 'function', 'method', 'unit', 'functionality', 'behavior', 'simple', 'ways',
                'capsule', 'capsules', 'medicines', 'details', 'project', 'language', 'word', 'text',
                'information', 'system', 'computer', 'software', 'development', 'program', 'application',
                'network', 'internet', 'web', 'technology', 'code', 'algorithm', 'process', 'user',
                'interface', 'file', 'database', 'server', 'client', 'hardware', 'device', 'input', 'output',
                'platform', 'environment', 'operating', 'system', 'website', 'service', 'protocol', 'interface',
                'communication', 'connection', 'address', 'router', 'switch', 'firewall', 'security',
                'encryption', 'authentication', 'authorization', 'session', 'session', 'management',
                'storage', 'memory', 'processor', 'CPU', 'GPU', 'RAM', 'ROM', 'cache', 'monitor', 'display',
                'keyboard', 'mouse', 'printer', 'scanner', 'backup', 'restore', 'archive', 'compression',
                'decompression', 'format', 'installation', 'upgrade', 'patch', 'update', 'version', 'release',
                'bug', 'error', 'exception', 'fault', 'failure', 'crash', 'debug', 'testing', 'quality',
                'assurance', 'verification', 'validation', 'documentation', 'manual', 'guide', 'tutorial',
                'video', 'lecture', 'course', 'lecture', 'workshop', 'conference', 'seminar', 'symposium',
                'meeting', 'event', 'agenda', 'schedule', 'calendar', 'planning', 'management', 'leadership',
                'team', 'project', 'task', 'deadline', 'milestone', 'goal', 'objective', 'requirement',
                'specification', 'design', 'architecture', 'framework', 'pattern', 'module', 'component',
                'library', 'package', 'dependency', 'versioning', 'integration', 'collaboration', 'cooperation',
                'communication', 'coordination', 'review', 'feedback', 'report', 'analysis', 'evaluation',
                'assessment', 'audit', 'survey', 'measurement', 'metric', 'indicator', 'index', 'benchmark',
                'standard', 'norm', 'best', 'practice', 'methodology', 'process', 'approach', 'strategy',
                'tactic', 'plan', 'schedule', 'timeline', 'budget', 'resource', 'allocation', 'utilization',
                'efficiency', 'productivity', 'performance', 'scalability', 'reliability', 'availability',
                'resilience', 'security', 'privacy', 'confidentiality', 'integrity', 'compliance', 'regulation',
                'standard', 'guideline', 'framework', 'model', 'concept', 'principle', 'paradigm', 'philosophy',
                'theory', 'hypothesis', 'assumption', 'axiom', 'law', 'rule', 'theorem', 'corollary', 'proof',
                'evidence', 'fact', 'truth', 'falsehood', 'belief', 'opinion', 'perspective', 'viewpoint',
                'interpretation', 'analysis', 'synthesis', 'argument', 'discussion', 'debate', 'dialogue',
                'conversation', 'interaction', 'communication', 'exchange', 'collaboration', 'cooperation',
                'teamwork', 'leadership', 'management', 'decision', 'making', 'problem', 'solving', 'creativity',
                'innovation', 'imagination', 'inspiration', 'motivation', 'passion', 'enthusiasm', 'dedication',
                'commitment', 'perseverance', 'persistence', 'patience', 'discipline', 'focus', 'concentration',
                'attention', 'awareness', 'consciousness', 'mindfulness', 'reflection', 'contemplation', 'meditation',
                'relaxation', 'stress', 'anxiety', 'depression', 'mental', 'health', 'wellbeing', 'fitness', 'exercise',
                'nutrition', 'diet', 'sleep', 'rest', 'recovery', 'balance', 'harmony', 'peace', 'serenity',
                'tranquility',
                'happiness', 'joy', 'fulfillment', 'contentment', 'satisfaction', 'success', 'achievement',
                'accomplishment',
                'progress', 'growth', 'development', 'evolution', 'change', 'adaptation', 'resilience', 'flexibility',
                'agility', 'mobility', 'speed', 'efficiency', 'effectiveness', 'optimization', 'performance',
                'productivity',
                'quality', 'excellence', 'superiority', 'competence', 'capability', 'capacity', 'resourcefulness',
                'creativity', 'innovation', 'ingenuity', 'entrepreneurship', 'initiative', 'leadership', 'organization',
                'administration', 'coordination', 'communication', 'collaboration', 'teamwork', 'problem', 'solving',
                'decision', 'making', 'critical', 'thinking', 'analytical', 'skills', 'problem', 'solving', 'decision',
                'making', 'creative', 'thinking', 'innovative', 'thinking', 'strategic', 'thinking', 'systems',
                'thinking', 'design', 'thinking', 'entrepreneurial', 'thinking', 'leadership', 'skills',
                'communication',
                'skills', 'presentation', 'skills', 'public', 'speaking', 'skills', 'writing', 'skills', 'listening',
                'skills', 'negotiation', 'skills', 'conflict', 'resolution', 'skills', 'team', 'building', 'skills',
                'time', 'management', 'skills', 'stress', 'management', 'skills', 'adaptability', 'skills',
                'flexibility',
                'skills', 'resilience', 'skills', 'emotional', 'intelligence', 'self', 'awareness', 'self',
                'management',
                'social', 'awareness', 'relationship', 'management', 'empathy', 'communication', 'skills',
                'assertiveness',
                'skills', 'leadership', 'styles', 'visionary', 'leadership', 'ethical', 'leadership', 'adaptive',
                'leadership', 'resilient', 'leadership', 'inclusive', 'leadership', 'distributed', 'leadership',
                'decentralized', 'leadership', 'collaborative', 'leadership', 'team', 'leadership', 'community',
                'devices', 'television', 'boxes'}

            self.stop_words = stop_words
            self.punctuation = punctuation
            self.additional_words = additional_words

        self.stoplist = self.stop_words | self.punctuation | self.additional_words
        return self.stoplist

    def is_proper_noun(self, word):
        import nltk
        if not self.nltk:
            self.lazy_import_nltk()

        words = nltk.word_tokenize(word)
        pos_tags = nltk.pos_tag(words)
        for _, tag in pos_tags:
            if tag == 'NNP':  # NNP stands for proper noun
                return True
        return False


    def get_requests(self):
        if not self.requests:
            import requests
            self.requests = requests
        return self.requests


    def tokenize_sentences(self, text):
        self.lazy_import_nltk()
        try:
            # Attempt to tokenize sentences
            return self.nltk.sent_tokenize(text)
        except LookupError:
            # If punkt resource is not found, download it
            self.nltk.download('punkt')
            # Retry sentence tokenization
            return self.nltk.sent_tokenize(text)





    def get_nouns_multipartite(self, text):
        out = set()
        stoplist = self.get_stoplist()  # Reuse stoplist

        words = self.nltk.word_tokenize(text)
        ner_tags = self.nltk.pos_tag(words)

        current_entity = ''
        for word, tag in ner_tags:
            if tag.startswith('NNP'):  # Check if the tag represents a proper noun
                current_entity += ' ' + word
            else:
                # Check if there's a current entity being built
                if current_entity.strip():
                    # Add the current entity to the output set
                    out.add(current_entity.strip())
                    # Reset the current entity
                    current_entity = ''
            # Check if the word is a noun or a number
            if tag.startswith('NN') or tag.startswith('CD'):
                if word.lower() not in stoplist and not word.isdigit():
                    if not (word.lower() in ['project', 'language']):
                        out.add(word)

        # Add any remaining entity to the output set
        if current_entity.strip():
            out.add(current_entity.strip())

        return out


    def get_wordsense(self, sent, word, preprocessed_word):
        # Check if result is in cache
        cache_key = (sent, word, preprocessed_word)
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Lazy import WordNet
        import nltk
        from nltk.corpus import wordnet as wn

        # Obtain synsets
        synsets = wn.synsets(preprocessed_word)

        if synsets:
            # Use the first synset assuming it's the most common meaning
            synset = synsets[0]

            # Initialize variables to store best similarity scores and corresponding synset
            best_score = -1
            best_synset = None

            # Calculate similarity for each synset
            for synset in synsets:
                word_synsets = wn.synsets(word)
                if word_synsets:
                    score = wn.wup_similarity(word_synsets[0], synset)
                    if score is not None and score > best_score:
                        best_score = score
                        best_synset = synset

            # Store only relevant information (best synset) in cache
            if len(self.cache) >= self.cache_size_limit:
                self.cache.popitem(last=False)  # Evict least recently used entry
            self.cache[cache_key] = best_synset
            return best_synset
        else:
            # Handle the case where the word is not found in WordNet
            # (e.g., print a warning or use a default score)
            print(f"Warning: '{preprocessed_word}' not found in WordNet.")
            # You can choose to return a default score or None here
            return None

    def get_distractors_wordnet(self, syn, preprocessed_word):
        hypernyms = syn.hypernyms()
        if hypernyms:
            hyponyms = hypernyms[0].hyponyms()
            for hyponym in hyponyms:
                name = hyponym.lemmas()[0].name().replace("_", " ")
                if name != preprocessed_word:
                    yield name.capitalize()



    def get_distractors_conceptnet(self, word):
        import requests  # Import requests module here
        if not self.requests:
            self.requests = requests

        preprocessed_word = word.lower().replace(" ", "_") if len(word.split()) > 0 else word.lower()
        original_word = preprocessed_word

        # Use NER to determine if the word is a proper noun (name)
        is_name = self.is_proper_noun(preprocessed_word)

        url = "http://api.conceptnet.io/query?node=/c/en/%s/n&rel=/r/PartOf&start=/c/en/%s&limit=5" % (
            preprocessed_word, preprocessed_word)
        obj = self.requests.get(url).json()
        for edge in obj['edges']:
            link = edge['end']['term']
            url2 = "http://api.conceptnet.io/query?node=%s&rel=/r/PartOf&end=%s&limit=10" % (link, link)
            obj2 = self.requests.get(url2).json()
            for edge in obj2['edges']:
                word2 = edge['start']['label']
                # If it's a name, return it
                if is_name:
                    yield word2
                # If it's not a name, return it only if it's not a common word
                else:
                    if word2.lower() not in self.stoplist:
                        yield word2


    def get_distractors_from_csv(self, input_file, keyword):
        encodings = ['latin-1']
        for encoding in encodings:
            try:
                with open(input_file, 'r', newline='', encoding=encoding) as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        key_concept = row['Key Concept']
                        distractors = row['Distractors'].split(', ')
                        if keyword.lower() in key_concept.lower() or keyword.lower() in distractors:
                            for distractor in distractors:
                                if distractor.lower() != keyword.lower():
                                    yield distractor
            except UnicodeDecodeError:
                print(f"Error decoding file with encoding {encoding}. Trying another encoding...")
        return []


    def get_sentences_for_keyword(self, keywords, sentences):
        keyword_sentence_mapping = {}
        for keyword in keywords:
            keyword_sentence_mapping[keyword] = []

        for sentence in sentences:
            for keyword in keywords:
                if keyword in sentence:
                    keyword_sentence_mapping[keyword].append(sentence)

        return keyword_sentence_mapping


    def get_distractors(self, keyword_sentence_mapping):
        key_distractor_list = {}
        for keyword in keyword_sentence_mapping:
            sentences = keyword_sentence_mapping[keyword]
            csv_distractors = list(self.get_distractors_from_csv('JAVA.csv', keyword))

            # Check if there are enough distractors from the CSV file
            if len(csv_distractors) >= 4:
                key_distractor_list[keyword] = csv_distractors
            else:
                wordsense = self.get_wordsense(sentences[0], keyword, preprocessed_word=keyword)
                if wordsense:
                    distractors = list(self.get_distractors_wordnet(wordsense, preprocessed_word=keyword))
                    if not distractors:
                        distractors = list(self.get_distractors_conceptnet(keyword))
                    if distractors:
                        # Combine distractors from CSV and other sources
                        combined_distractors = csv_distractors + distractors
                        key_distractor_list[keyword] = combined_distractors[:4]  # Take the first 4 distractors

        return key_distractor_list


    def generate_mcqs(self, text_data):
        sentences = self.tokenize_sentences(text_data)
        keywords = self.get_nouns_multipartite(text_data)
        keyword_sentence_mapping = self.get_sentences_for_keyword(keywords, sentences)
        key_distractor_list = self.get_distractors(keyword_sentence_mapping)
        return self.generate_mcqs_from_data(keyword_sentence_mapping, key_distractor_list)

    def generate_mcqs_from_data(self, keyword_sentence_mapping, key_distractor_list):
        import re
        import random

        option_choices = ['a', 'b', 'c', 'd']

        for keyword in key_distractor_list:
            sentences = keyword_sentence_mapping[keyword]
            if sentences:
                sentence = sentences[0]
                pattern = re.compile(re.escape(keyword), re.IGNORECASE)
                output = pattern.sub(" _______ ", sentence)

                distractors = list(key_distractor_list[keyword])

                if keyword not in distractors:
                    distractors.append(keyword)

                distractors = random.sample(distractors, 3)  # Always select 3 distractors

                distractors += [''] * (4 - len(distractors))

                # Shuffle the distractors again to randomize their order
                random.shuffle(distractors)

                mcq = {"question": output, "answer": keyword, "options": dict(zip(option_choices, distractors))}
                yield mcq

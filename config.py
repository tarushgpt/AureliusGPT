import os

#preprocessing 
greekwords = ["ἔμφρων", "σύμφρων", "ὑπέρφρων", "Νόμος", "νέμων","ἀξιοπίστως","θεοφόρητος", "οἰκονομίαν", "τοῦτο", "ἔφερεν", "αὐτῷ", "εὔμοιρος", "εὐδαιμονία", "εὐπατρίδαι", "καθότι", "κατορθώσεως", "κόσμος", "μέλος", "μέρος", "παρειλήφαμεν", "συμβαίνειν", "τάσις", "ἀγαθός", "ἀκτῖνες", "ἐκτείνεσθαι", "δαίμων", "κατορθώσεις", "ἀγαθὸς", "ἀυτῷ"]
greek_transliteration = ["emphron", "sumphron", "huperphron", "nomos", "nemon", "axiopistos", "theophoretos", "oikonomian", "touto", "eferen", "auto", "eumoiros", "eudaimonia", "eupatridai", "kathoti", "katorthoseos", "kosmos", "melos", "meros", "pareilephamen", "symbainein", "tasis", "agathos", "aktines", "ekteinesthai", "daimon", "katorthoseis", "agathos", "auto"]



d_model = 256
d_k = 256
d_v = 256
h = 4
d_ff = 1024
max_seq_length = 512
lr = 0.001
num_blocks = 3
vocab_length = 1000
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
min_freq = 30
n = 10000
epsilon = 10**-8
max_tokens_inference = 50
temperature = 1
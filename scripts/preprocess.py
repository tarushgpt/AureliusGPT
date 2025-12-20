import re, os, unicodedata

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def preprocess(string):
    meditations = unicodedata.normalize("NFC", string)

    startindex = meditations.index("THE FIRST BOOK")
    meditations = meditations[startindex:]
    endindex = meditations.index("APPENDIX")
    meditations = meditations[:endindex] 

    book_name = r"THE [A-Z]+ BOOK"
    section_name = r"\n[IVXLCDM]+\. "
    whitespace = r"[\n]+"
    underline = r"[_]+"
    asterix = r"[*]+"
    ae = r"[œ]"


    meditations = re.sub(book_name, "", meditations)
    meditations = re.sub(whitespace, "\n", meditations)
    meditations = re.sub(section_name, "\n\n", meditations)
    meditations = re.sub(underline, "", meditations)
    meditations = re.sub(asterix, "", meditations)
    meditations = re.sub(ae, "ae", meditations)

    greekwords = ["ἔμφρων", "σύμφρων", "ὑπέρφρων", "Νόμος", "νέμων","ἀξιοπίστως","θεοφόρητος", "οἰκονομίαν", "τοῦτο", "ἔφερεν", "αὐτῷ", "εὔμοιρος", "εὐδαιμονία", "εὐπατρίδαι", "καθότι", "κατορθώσεως", "κόσμος", "μέλος", "μέρος", "παρειλήφαμεν", "συμβαίνειν", "τάσις", "ἀγαθός", "ἀκτῖνες", "ἐκτείνεσθαι", "δαίμων", "κατορθώσεις", "ἀγαθὸς", "ἀυτῷ"]
    greek_transliteration = ["emphron", "sumphron", "huperphron", "nomos", "nemon", "axiopistos", "theophoretos", "oikonomian", "touto", "eferen", "auto", "eumoiros", "eudaimonia", "eupatridai", "kathoti", "katorthoseos", "kosmos", "melos", "meros", "pareilephamen", "symbainein", "tasis", "agathos", "aktines", "ekteinesthai", "daimon", "katorthoseis", "agathos", "auto"]


    for i in range(len(greekwords)):
        meditations = meditations.replace(greekwords[i], greek_transliteration[i])

    return meditations


def main():

    with open(os.path.join(PROJECT_ROOT,"data/raw/meditations.txt"), "r") as f:
            meditations = f.read()

    meditations = preprocess(meditations)

    with open(os.path.join(PROJECT_ROOT,"data/processed/meditations.txt"), "w") as f:
            try:
                f.write(meditations)
                print("Successfully wrote Meditations.")
            except Exception as e:
                print(e)


if __name__ == '__main__':
    main()
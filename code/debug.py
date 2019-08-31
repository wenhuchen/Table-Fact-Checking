import pandas
from beam_search import dynamic_programming
import spacy

nlp = spacy.load('en_core_web_sm')

if __name__ == "__main__":
    table = 3
    if table == 1:
        t = pandas.read_csv('../data/all_csv/1-1341423-13.html.csv', delimiter="#")
    elif table == 2:
        t = pandas.read_csv('../data/all_csv/2-10808089-16.html.csv', delimiter="#")
    elif table == 3:
        t = pandas.read_csv('../data/all_csv/1-28498999-6.html.csv', delimiter="#")
    else:
        pass

    cols = t.columns
    cols = cols.map(lambda x: x.replace(' ', '_') if isinstance(x, (str, unicode)) else x)
    t.columns = cols
    print t
    option = -1

    if option == 1:
        sent = u"bill lipinski and luis gutierrez are both democratic"
        tags = [_.tag_ for _ in nlp(sent)]
        mem_str = [('party', 'democratic'), ('incumbent', 'bill lipinski'), ('incumbent', 'luis gutierrez')]
        head_str = []
        mem_num = []
        head_num = []
    elif option == 2:
        sent = u'there are 3 _ _ _ in _'
        tags = [_.tag_ for _ in nlp(sent)]
        mem_str = []
        head_str = ['incumbent']
        mem_num = [('first_elected', 1998), ("tmp_none", 3)]
        head_num = ['first_elected']
    elif option == 3:
        sent = u'phi crane is _ with the earliest year of _'
        tags = [_.tag_ for _ in nlp(sent)]
        mem_str = [('incumbent', 'phil crane')]
        head_str = ['incumbent']
        mem_num = []
        head_num = ['first_elected']
    elif option == 4:
        sent = u"there is more _ oriented incumbents than _"
        tags = [_.tag_ for _ in nlp(sent)]
        mem_str = [('party', 'democratic'), ('party', 'republican')]
        head_str = []
        mem_num = []
        head_num = []
    elif option == -1:
        sent = u"luke donald is not one of the two who had 24 events."
        tags = [_.tag_ for _ in nlp(sent)]
        mem_str = [('player', 'luke donald')]
        head_str = ['player']
        mem_num = [("tmp_none", 2), ("events", 24)]
        head_num = ["events"]
    elif option == -2:
        sent = u'united states happens more times than any other teams'
        tags = [_.tag_ for _ in nlp(sent)]
        mem_str = [('country', 'united states')]
        head_str = ['country']
        mem_num = []
        head_num = []
    elif option == 5:
        sent = u'there are 2 _ who were not _'
        tags = [_.tag_ for _ in nlp(sent)]
        mem_str = [("results", "re - elected")]
        head_str = ['incumbent']
        mem_num = [("tmp_none", 2)]
        head_num = []
    elif option == 6:
        sent = u'all _ are _ _'
        tags = [_.tag_ for _ in nlp(sent)]
        mem_str = [("results", "re - elected")]
        head_str = ['incumbent']
        mem_num = []
        head_num = []
    elif option == 7:
        sent = u'the earliest _ is _ in _'
        tags = [_.tag_ for _ in nlp(sent)]
        mem_str = []
        head_str = ['incumbent']
        mem_num = [('first_elected', 1994)]
        head_num = ['first_elected']
    elif option == 8:
        sent = u'_ _ _ are all _'
        tags = [_.tag_ for _ in nlp(sent)]
        mem_str = [('incumbent', 'danny k. davis'), ('incumbent', 'lane evans'), ('party', 'republican')]
        head_str = []
        mem_num = []
        head_num = []
    elif option == 9:
        sent = u'st kilda lost to essendon and hawthorn lost to south melbourne'
        tags = [_.tag_ for _ in nlp(sent)]
        mem_str = [('home_team', 'st kilda'), ('away_team', 'south melbourne'), ('home_team', 'collingwood'), ('away_team', 'north melbourne')]
        head_str = []
        mem_num = []
        head_num = []
    elif option == 10:
        sent = u'The game with the fewest number of people in attendance was hawthorn vs south melbourne'
        tags = [_.tag_ for _ in nlp(sent)]
        mem_str = [('home_team', 'hawthorn'), ('away_team', 'footscray')]
        head_str = []
        mem_num = []
        head_num = ['crowd']
    elif option == 11:
        sent = u'collingwood is following essendon'
        tags = [_.tag_ for _ in nlp(sent)]
        mem_str = [('home_team', 'collingwood'), ('home_team', 'essendon')]
        head_str = ['home_team']
        mem_num = []       
        head_num = []

    dynamic_programming(t, sent, tags, mem_str, mem_num, head_str, head_num, 6)
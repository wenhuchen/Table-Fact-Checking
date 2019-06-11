This file is aimed to explain the data format.

all_csv.zip is the zipped file for all the tables, there are totally 16K tables in the folder. All the csv files are separated with '#' separator.

full_cleaned.json is the file consisting of all the data, which has been cleaned and pre-processed with entity linking. A typical data frame looks like this:

"2-1859269-1.html.csv": [
    [
      "during the #third round;c0# of the #turkish cup;c-1# , there be no #new entries;h4# during that #stage;c0#",
      "the highest number of #winners from;h3# a #previous round;h3# in the #turkish cup;c-1# be #54;c2# in #round;h0# 3",
      "#s\u00fcper lig;c5# be the most common leagues to win a #round;h0# in the #turkish cup;c-1#",
      "the lowest number of #new entries;h4# conclude a #round;h0# in the #turkish cup;c-1# be #5;c4#",
      "#round;h0# 1 of the #turkish cup;c-1# begin with #156;c1# competitor and the #finals;c0# #round;h0# only complete with #2;c1#",
      "there be #new entries;h4# for the 1st #4;c3# #round;h0# of the #turkish cup;c-1#",
      "the highest number of #winners from;h3# a #previous round;h3# in the urkish #cup;c-1# be #59;c1# in #round;h0# 3",
      "#tff third leagues;c5# be the most common leagues to win a #round;h0# in the #turkish cup;c-1#",
      "#2;c1# be the lowest number of #new entries;h4# conclude a #round;h0# in the #turkish cup;c-1#",
      "#from round;h3# 1 to the #finals;c0# #round;h0# , there be #4;c3# #clubs remaining;h1# to complete the #round;h0#"
    ],
    [
      1,
      1,
      1,
      1,
      1,
      0,
      0,
      0,
      0,
      0
    ],
    [
      "IN DT ENT IN DT ENT , EX VBD DT ENT IN DT ENT",
      "DT JJS NN IN ENT DT ENT IN DT ENT VBD ENT IN ENT CD",
      "ENT VBD DT RBS JJ ENT TO VB DT ENT IN DT ENT",
      "DT JJS NN IN ENT VBG DT ENT IN DT ENT VBD ENT",
      "ENT CD IN DT ENT VBD IN ENT NNS CC DT ENT ENT RB VBD IN ENT",
      "EX VBD ENT IN DT CD ENT ENT IN DT ENT",
      "DT JJS NN IN ENT DT ENT IN DT JJ ENT VBD ENT IN ENT CD",
      "ENT VBD DT RBS JJ ENT TO VB DT ENT IN DT ENT",
      "ENT VBD DT JJS NN IN ENT VBG DT ENT IN DT ENT",
      "ENT CD TO DT ENT ENT , EX VBD ENT ENT TO VB DT ENT"
    ],
    "turkish cup"
  ]

The key is the table file name, the list contains four elements:
- first element contains all the statements, the ## are used enclose the linked entity, #;c0# means the entity links to some table cell in the first column, #;c-1# means the entity links to the table caption. 
- second element contains all the labels for the corresponding statement, where 1 denotes entailed and 0 denotes refuted.
- third element contains the pos tag of the statement, the linked entity shares the same tag <ENT>.
- fourth element is the caption of the table.

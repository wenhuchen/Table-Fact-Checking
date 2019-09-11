full_cleaned.json is the file consisting of all the data, which has been cleaned and pre-processed with entity linking. A typical data frame looks like this:
```
"2-1859269-1.html.csv": [
    [
      "during the #third round;0,2# of the #turkish cup;-1,-1# , there be no #new entries;4,3# during that #stage;0,3#",
    ],
    [
      1,
    ],
    [
      "IN DT ENT IN DT ENT , EX VBD DT ENT IN DT ENT",
    ],
    "turkish cup"
  ]
```
The key is the table file name, the list contains four elements:
- first element contains all the statements, the ## are used enclose the linked entity, #;3,4# means the entity links to some table cell in the third row in the fourth column.

- second element contains all the labels for the corresponding statement, where 1 denotes entailed and 0 denotes refuted.

- third element contains the pos tag of the statement, the linked entity shares the same tag <ENT>.
    
- fourth element is the caption of the table.

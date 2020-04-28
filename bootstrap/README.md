The data format of bootstrap.json is described below:

```
{
  "2-11545282-3.html.csv": [
    [
      "jarron collins played for the jazz for longer than dell curry did .",
      null,
      "utah jazz all - time roster",
      null,
      "greater{hop{filter_eq{all_rows; player; jarron collins}; years for jazz}; hop{filter_eq{all_rows; player; dell curry}; years for jazz}}=True"
    ]
  ],
  "1-28677723-14.html.csv": [
    [
      "for season 6 of skal vi danse?, when the total was over 30, there were two times that the style was jive .",
      null,
      "skal vi danse? (season 6)",
      null,
      "eq{count{filter_eq{filter_greater{all_rows; total; 30}; style; jive}}; 2}=True"
    ]
  ],
  ...
}
```
Each entry has its key as the table name, the value contains the original sentence, table caption and the program. Some definitions of these functions are listed as below:
```
greater(A, B): A is greater than B, return True, other return False
hop(Row, Field Name): Hop to the Field name column in the Row.
count(C): Counting how many rows are in the given C Rows.
...
```


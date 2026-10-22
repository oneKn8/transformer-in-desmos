## Grid

| config | params | full: 32-blocks | full: elementwise | ms/token full (padded / mac) | ms/token kv (padded / mac) | steps | val bpc (all) | val bpc (ctx>=T/2) | test bpc (ctx>=T/2) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| d32_L1_H2_T32 | 15,456 | 17 | 21,696 | 94 / 90 | 75 / 54 | 1000 | 2.318 | 2.258 | 2.254 |
| d32_L1_H4_T32 | 15,456 | 21 | 27,840 | 105 / 96 | 80 / 54 | 1000 | 2.308 | 2.249 | 2.236 |
| d32_L2_H2_T32 | 27,808 | 33 | 42,176 | 135 / 129 | 98 / 57 | 1000 | 2.238 | 2.176 | 2.164 |
| d64_L1_H2_T32 | 55,488 | 54 | 28,928 | 149 / 147 | 125 / 57 | 1000 | 2.179 | 2.114 | 2.099 |
| d32_L2_H4_T32 | 27,808 | 41 | 54,464 | 158 / 141 | 109 / 57 | 1000 | 2.225 | 2.155 | 2.146 |
| d32_L1_H2_T64 | 16,480 | 41 | 55,488 | 159 / 147 | 82 / 56 | 1000 | 2.308 | 2.276 | 2.265 |
| d64_L1_H4_T32 | 55,488 | 58 | 43,264 | 169 / 161 | 131 / 58 | 1000 | 2.176 | 2.108 | 2.096 |
| d32_L1_H4_T64 | 16,480 | 57 | 80,064 | 204 / 172 | 93 / 56 | 1000 | 2.281 | 2.243 | 2.239 |
| d32_L4_H2_T32 | 52,512 | 65 | 83,136 | 218 / 206 | 145 / 63 | 1000 | 2.175 | 2.100 | 2.087 |
| d64_L2_H2_T32 | 104,768 | 106 | 55,552 | 243 / 241 | 198 / 64 | 1000 | 2.101 | 2.027 | 2.012 |
| d32_L4_H4_T32 | 52,512 | 81 | 107,712 | 263 / 231 | 167 / 64 | 1000 | 2.168 | 2.091 | 2.078 |
| d32_L2_H2_T64 | 28,832 | 81 | 108,736 | 264 / 242 | 113 / 61 | 1000 | 2.232 | 2.199 | 2.192 |
| d64_L1_H2_T64 | 57,536 | 114 | 69,888 | 268 / 266 | 135 / 62 | 1000 | 2.191 | 2.161 | 2.154 |
| d64_L2_H4_T32 | 104,768 | 114 | 84,224 | 282 / 270 | 209 / 65 | 1000 | 2.105 | 2.029 | 2.019 |
| d64_L1_H4_T64 | 57,536 | 130 | 110,848 | 330 / 307 | 146 / 63 | - | not run | not run | not run |
| d32_L2_H4_T64 | 28,832 | 113 | 157,888 | 355 / 291 | 135 / 62 | - | not run | not run | not run |
| d64_L4_H2_T32 | 203,328 | 210 | 108,800 | 432 / 429 | 343 / 79 | - | not run | not run | not run |
| d32_L4_H2_T64 | 53,536 | 161 | 215,232 | 475 / 432 | 175 / 73 | - | not run | not run | not run |
| d64_L2_H2_T64 | 106,816 | 226 | 135,424 | 479 / 477 | 217 / 73 | - | not run | not run | not run |
| d64_L4_H4_T32 | 203,328 | 226 | 166,144 | 510 / 487 | 366 / 80 | - | not run | not run | not run |
| d64_L2_H4_T64 | 106,816 | 258 | 217,344 | 603 / 559 | 239 / 75 | - | not run | not run | not run |
| d32_L4_H4_T64 | 53,536 | 225 | 313,536 | 656 / 530 | 218 / 74 | - | not run | not run | not run |
| d64_L4_H2_T64 | 205,376 | 450 | 266,496 | 901 / 899 | 381 / 96 | - | not run | not run | not run |
| d64_L4_H4_T64 | 205,376 | 514 | 430,336 | 1149 / 1063 | 425 / 99 | - | not run | not run | not run |

## Pareto frontier (val bpc vs padded ms/token)

| config | ms/token full (padded) | val bpc (ctx>=T/2) | params |
|---|---:|---:|---:|
| d32_L1_H2_T32 | 94 | 2.258 | 15,456 |
| d32_L1_H4_T32 | 105 | 2.249 | 15,456 |
| d32_L2_H2_T32 | 135 | 2.176 | 27,808 |
| d64_L1_H2_T32 | 149 | 2.114 | 55,488 |
| d64_L1_H4_T32 | 169 | 2.108 | 55,488 |
| d32_L4_H2_T32 | 218 | 2.100 | 52,512 |
| d64_L2_H2_T32 | 243 | 2.027 | 104,768 |

## Ablations around d64_L2_H2_T32

| variant | params | ms/token full (padded) | val bpc (ctx>=T/2) | delta vs base | test bpc (ctx>=T/2) |
|---|---:|---:|---:|---:|---:|
| base (d64_L2_H2_T32) | 104,768 | 243 | 2.027 | 0 | 2.012 |
| act-relu | 104,768 | 243 | 2.014 | -0.013 | 2.004 |
| bias | 105,952 | 280 | 2.025 | -0.002 | 2.009 |
| norm-ln | 105,088 | 260 | 2.031 | +0.004 | 2.011 |
| norm-none | 104,448 | 227 | 2.165 | +0.138 | 2.157 |
| pos-alibi | 102,720 | 243 | 2.327 | +0.300 | 2.315 |
| seed1 | 104,768 | 243 | 2.006 | -0.022 | 1.993 |
| tied | 102,720 | 243 | 2.040 | +0.013 | 2.029 |
| v64 | 108,864 | 246 | 2.060 | +0.033 | 2.035 |

## Homer vs tiny-shakespeare (1000 steps each)

| config | ms/token full (padded) | Homer val / test bpc (ctx>=T/2) | Shakespeare val / test bpc (ctx>=T/2) |
|---|---:|---:|---:|
| d32_L4_H2_T32 | 218 | 2.100 / 2.087 | 2.410 / 2.529 |
| d64_L1_H2_T32 | 149 | 2.114 / 2.099 | 2.498 / 2.562 |
| d64_L2_H2_T32 | 243 | 2.027 / 2.012 | 2.358 / 2.471 |

## Corpus samples

**d32_L4_H2_T32, Homer** (val bpc 2.100):

```
then minerva said, "the ships why shire she were tearever fair fight the end of against sating, and at toldians, who broe in they upon of the was in as the bed sogating to she hoursed it wine, but seat aloud, "they were
```

**d32_L4_H2_T32, Shakespeare** (val bpc 2.410):

```
romeo:
if you revenge threasign ive mid
haut anchied have beady your: if in desconderieven,
in him stuppparlant.

marcius:
i come, and your for away if will lear from,
be it spray the hate for to so the vist
```

**d64_L1_H2_T32, Homer** (val bpc 2.114):

```
then minerva said, "the ships who havings of we stearever of seating the enten they through are suitor," sons to came the was towarder of the was i am the sons still had answered, and it wine upon seat aloud, "cargues t
```

**d64_L1_H2_T32, Shakespeare** (val bpc 2.498):

```
romeo:
o shall it know murdering vices ned head head,
and that sun your severy, you sight this havest,
and i unworld, this do, my und my from by wherefore accusinfering hereis rometh:
here for that be fair l
```

**d64_L2_H2_T32, Homer** (val bpc 2.027):

```
then minerva said, "the ships who was gashed were eared abodis and with went of all, these was from and joved them on the would with fight at will have you bed still had answered, and it wine upon seat aloud, "into as t
```

**d64_L2_H2_T32, Shakespeare** (val bpc 2.358):

```
romeo:
o shall the great him go in this haught chrief,
but mades, you make know you sights,
in himself that a madam uptain, but seem thy frown yearn's good
yet inferces new is fair apport
in the as of fair l
```


## Learning-rate check

| run | lr | steps | val bpc (ctx>=T/2) |
|---|---:|---:|---:|
| lr1e-2_d32_L2_H4_T32 | 0.01 | 1000 | 2.187 |
| lr1e-2_d64_L2_H4_T32 | 0.01 | 1000 | 2.029 |
| lr1e-3_d32_L2_H4_T32 | 0.001 | 1000 | 2.678 |
| lr2e-2_d32_L2_H4_T32 | 0.02 | 1000 | 2.155 |
| lr3e-3_d32_L2_H4_T32 | 0.003 | 1000 | 2.395 |
| lr3e-3_d64_L2_H4_T32 | 0.003 | 1000 | 2.088 |

## Samples, best configs at <= 300 ms/token

**d64_L2_H2_T32** (val bpc 2.027, ~243 ms/token):

```
then minerva said, "the ships who was gashed were eared abodis and with went of all, these was from and joved them on the would with fight at will have you bed still had answered, and it wine upon seat aloud, "into as t
```

**d64_L2_H4_T32** (val bpc 2.029, ~282 ms/token):

```
then minerva said, "my done wall that has care and afe i wed seent warriends; for your feasirped you the son off their her of the hourder, and serva as the bed still had answered, and it wine, but seat they were tolder 
```

**d32_L4_H4_T32** (val bpc 2.091, ~263 ms/token):

```
then minerva said, "the ships who has gashed we steared about not his shoef." far your leath over hear," son phus and she would were field at will a as you bed son of a made greatts for the fire opleas and crew at by in
```

**d32_L4_H2_T32** (val bpc 2.100, ~218 ms/token):

```
then minerva said, "the ships why shire she were tearever fair fight the end of against sating, and at toldians, who broe in they upon of the was in as the bed sogating to she hoursed it wine, but seat aloud, "they were
```


## Samples, best configs overall

**d64_L2_H2_T32** (val bpc 2.027, ~243 ms/token):

```
then minerva said, "the ships who was gashed were eared abodis and with went of all, these was from and joved them on the would with fight at will have you bed still had answered, and it wine upon seat aloud, "into as t
```

**d64_L2_H4_T32** (val bpc 2.029, ~282 ms/token):

```
then minerva said, "my done wall that has care and afe i wed seent warriends; for your feasirped you the son off their her of the hourder, and serva as the bed still had answered, and it wine, but seat they were tolder 
```

**d32_L4_H4_T32** (val bpc 2.091, ~263 ms/token):

```
then minerva said, "the ships who has gashed we steared about not his shoef." far your leath over hear," son phus and she would were field at will a as you bed son of a made greatts for the fire opleas and crew at by in
```

**d32_L4_H2_T32** (val bpc 2.100, ~218 ms/token):

```
then minerva said, "the ships why shire she were tearever fair fight the end of against sating, and at toldians, who broe in they upon of the was in as the bed sogating to she hoursed it wine, but seat aloud, "they were
```


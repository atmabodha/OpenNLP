# Fictometer

## Description

Fictometer is an algorithm for analysing whether the given text is ```Fiction``` or ```Non-Fiction```.
It first calculates the number of ```adverbs```, ```adjectives``` and ```pronouns``` in the text.
It then calculates Ratio of Adjective to Pronoun ```RADJPRON``` and Ratio of Adverb to Adjective ```RADVADJ```, 
from which it predicts whether text is ```Fiction``` or ```Non-Fiction```.

Blog Link: 🔗[LINK](https://bekushal.medium.com/fictometer-a-simple-and-explainable-algorithm-for-sentiment-analysis-31186d2a8c7e)


## Installation

```bash
pip install Fictometer
```


## Usage

    import Fictometer

    text = "your_text"

    pc = Fictometer.ficto.pos_count(text)
    pr = Fictometer.ficto.pos_ratio(pc)
    result = Fictometer.ficto.fictometer(pr)
    Fictometer.ficto.help()


## Contact

email - atmabodha@gmail.com
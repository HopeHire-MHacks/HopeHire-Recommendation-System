# HopeHire Recommendation System

HopeHire uses a recommendation system aims to better serve both kidney patients and employers by recommending potential employment opportunities between the two parties.

## Implementation

#### Empowered by [Tensorflow Recommenders](https://www.tensorflow.org/recommenders)

## Description

By using a 2 step proccess of _**Retrieval**_ then _**Ranking**_ approach enables efficient computation efficiency and ranking.  
We have **2 retrieval models** to enable the retrieval of potential employers to kidney patients and retrieval of potential employees to employers.

The networks are trained on context features of both the kidney patients and employers such as age, skills and location to produce a more accurate prediction.

## Technology

|||
|-:|:-|
| AI | Tensorflow, Tensorflow-Recommenders |
| Serving | Flask |

## Future Enhancements

- Continuous training and deployment of recommendation networks
- Additional features used for networks
- Finetuning and improving architecture of networks
# MediaEval 21 - Driving Road Safety Forward: Video Data Privacy

## Why do some drivers crash?

The lifetime odds for dying in a car crash are <b>1 in 103</b>. Each year, vehicle crashes cost <b>hundreds of billions</b> of dollars.

Research shows that <b> driver behavior</b> is key to automobile accidents. It’s a primary factor in ⅔ of crashes and a contributing factor in 90% of crashes. However, driver behavior is one of the most difficult things to study.

 <b>In-cabin video data</b> has the potential to enhance our understanding and provide greater context for interpretation, but brings forward important questions about responsible data science and the future of research. 

## The Task

The goal of this video data task is to explore methods for obscuring  driver identity in driver-facing video recordings while preserving human  behavioral information.

The <b>SHRP 2 dataset</b> was collected as part of the <b>Naturalistic Driving Study (NDS)</b> and contains <b> millions of hours and petabytes of driver video data</b> that can be used by researchers to gain a better understanding about the underlying causes of car crashes.

The dataset is currently <b> hosted in a secure enclave</b> due to privacy concerns about the identities of the drivers used in the study. The data studied in this challenge is similar to the SHRP 2 dataset but the privacy concerns and participant submissions are applicable in a much broader context of the driver safety research community.

## [Data](DATA.md)
The data folder contains the dataset to be used for the competition.

## Evaluation methodology

The evaluation process includes a preliminary automated evaluation as well as a human evaluation, to assess the de-identification of faces and measure the consistency in preserving driver actions and emotions. An initial automated process will be run using a deep learning-based gaze estimator. The difference in predicted gaze-vectors from the original un-filtered video and de-identified video will be used as an initial score. Human evaluators will use the evaluation methodology as described by Baragchizadeh et al. in Evaluation of Automated Identity Masking Method (AIM) in Naturalistic Driving Study (NDS) [9].

The scores for each of these areas will be combined for an overall assessment, prioritizing the human assessment of de-identification. PLEASE NOTE that this Task is heavily reliant on human evaluation, and we encourage participants to include in their submission any ideas, methods, and results from their own evaluation approaches. The participants’ descriptions of methodology, assumptions, and results will be shared with reviewers and the project organizers for additional discussion and opportunities for seed funding for further research.

Although we encourage all Task participants to think creatively and holistically about how the expectations of privacy, the risk from potential attackers, and various threat models may evolve, our starting assumptions are that: (1) The drivers are not known to the potential attacker. We assume there is no relationship between the attacker and the driver. Furthermore, it is assumed that the driver is not a public figure. (2) Any information from the driver’s surroundings is assumed to not influence the attacker’s ability to identify the driver. (3) Access to the data is limited to registered users who have signed a Data Use Agreement specifying they will not attempt to learn the identity of individuals in the videos. (4) Attackers have access to basic computational resources. (5) There is a low probability of attackers launching an effective crowdsourcing strategy to re-identify the drivers, in part due to the Data Use Agreement and context in which the data were collected.

The organizers of this Task encourage open source code with a MIT license, and the open sharing of insights to support a multidisciplinary community of practice. We anticipate that with the engagement of the MediaEval community there will be multiple opportunities to highlight both quantitative and qualitative feedback from participants, supporting reproducibility, open science, and future collaborative research.

- [6] Finch, K. (2016, April 25). A visual guide to practical data de-identification. Retrieved March 28, 2021, from https://fpf.org/blog/a-visual-guide-to-practical-data-de-identification/

- [8] Ferrell, R., Aykac, D., Karnowski, T., & Srinivas, N. (2021, January). A Publicly Available, Annotated Data Set for Naturalistic Driving Study and Computer Vision Algorithm Development. Retrieved from https://info.ornl.gov/sites/publications/Files/Pub122418.pdf

- [9] Baragchizadeh, Asal, O’Toole, Alice, Karnowski, Thomas Paul, & Bolme, David S. Evaluation of Automated Identity Masking Method (AIM) in Naturalistic Driving Study (NDS). United States. https://doi.org/10.1109/FG.2017.54

## Task Schedule

- July 2021: Registration on Submittable opens
- July 2021: Data release to registered participants
- August-October 2021: Community webinars/mentoring
- October 2021: Runs due
- November 2021: Results returned
- 22 November 2021: Working notes paper
- 6-8 December 2021: MediaEval 2021 Workshop
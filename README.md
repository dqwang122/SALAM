# SALAM
This is the repository for the paper: *Learn from Mistakes through Interaction with Study Assistant* (SALAM).
This project is still on process. We will update the code and data soon.

![SALAM](figs/framework.pdf)

## Data
The benchmarks we used in the paper are available in the `data` folder. The data is in the format of JSON. They are borrowed from the following repositories:

* Big Bench Hard (BBH): https://github.com/suzgunmirac/BIG-Bench-Hard
* Bias Benchmark for QA (BBQ): https://github.com/nyu-mll/BBQ

For BBH, we select 16 English multi-choice tasks. For BBQ, we randomly select 250 examples for each task.



## Code

Some code was borrowed from the following repositories. We thank the authors for their great work!
* Quark: https://github.com/GXimingLu/Quark
* Manifest: https://github.com/HazyResearch/manifest
* revChatGPT: https://github.com/acheong08/ChatGPT


## Results

### Overall results on two benchmarks.

|          |      |  BBH  |       |       |   BBQ    |       |
| -------- | ---- | ---- |  ---- |  ---- |  ---- |  ---- | 
|          | Min  | Max  | Average | Min | Max | Average |
| Original | 10.0 | 72.0 | 42.4 | 62.0 | 86.0 | 76.6 |
| Correct  | 0.0  | 84.0 | 38.4 | 60.0 | 88.0 | 72.0 |
| Mistake  | 0.0  | 84.0 | 37.9 | 72.0 | 90.0 | 79.8 |
| Feedback | 10.0 | 84.0 | 47.1 | 80.0 | 96.0 | 85.3 |


### Detail Results on BBQ
|                     | Original | Correct | Mistake | Feedback |
| ------------------- | -------- | ------- | ------- | -------- |
| Age                 | 68.0     | 68.0    | 72.0    | 96.0     |
| Disability_status   | 62.0     | 62.0    | 72.0    | 80.0     |
| Gender_identity     | 84.0     | 68.0    | 82.0    | 82.0     |
| Nationality         | 76.0     | 84.0    | 82.0    | 82.0     |
| Physical_appearance | 74.0     | 60.0    | 74.0    | 82.0     |
| Race_ethnicity      | 84.0     | 88.0    | 82.0    | 86.0     |
| Race_x_SES          | 86.0     | 76.0    | 84.0    | 86.0     |
| Race_x_gender       | 74.0     | 76.0    | 84.0    | 82.0     |
| Religion            | 82.0     | 66.0    | 80.0    | 82.0     |
| SES                 | 82.0     | 80.0    | 90.0    | 92.0     |
| Sexual_orientation  | 70.0     | 64.0    | 76.0    | 88.0     |
| Avg                 | 76.55    | 72.00   | 79.82   | 85.27    |


### Detail Results on BBH
|                                         | Original | Correct | Mistake | Feedback |
| --------------------------------------- | -------- | ------- | ------- | -------- |
| date_understanding                      | 48.0     | 48.0    | 46.0    | 46.0     |
| disambiguation_qa                       | 64.0     | 68.0    | 70.0    | 80.0     |
| geometric_shapes                        | 14.0     | 12.0    | 6.0     | 14.0     |
| hyperbaton                              | 62.0     | 84.0    | 84.0    | 84.0     |
| logical_deduction_five_objects          | 50.0     | 20.0    | 40.0    | 70.0     |
| logical_deduction_seven_objects         | 64.0     | 4.0     | 6.0     | 62.0     |
| logical_deduction_three_objects         | 72.0     | 78.0    | 58.0    | 72.0     |
| movie_recommendation                    | 30.0     | 54.0    | 44.0    | 42.0     |
| penguins_in_a_table                     | 46.7     | 16.7    | 16.7    | 43.3     |
| reasoning_about_colored_objects         | 62.0     | 60.0    | 62.0    | 64.0     |
| ruin_names                              | 16.0     | 22.0    | 28.0    | 26.0     |
| snarks                                  | 61.1     | 77.8    | 75.0    | 75.0     |
| temporal_sequences                      | 26.0     | 28.0    | 24.0    | 26.0     |
| tracking_shuffled_objects_five_objects  | 18.0     | 14.0    | 18.0    | 10.0     |
| tracking_shuffled_objects_seven_objects | 10.0     | 0.0     | 0.0     | 16.0     |
| tracking_shuffled_objects_three_objects | 34.0     | 28.0    | 28.0    | 24.0     |
| Average                                 | 42.4     | 38.4    | 37.9    | 47.1     |
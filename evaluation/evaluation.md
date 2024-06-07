# Self-Reflection Evaluation

The evaluation process assessed the model's ability to generate accurate and relevant responses by testing it with questions and corresponding contexts. Responses were generated and evaluated based on their relevance, grounding in the provided context, and overall utility. 

<h4>1- Prompt and Context Formatting:</h4>  Each evaluation instance began with a specific question (prompt) and, where applicable, relevant contextual information. The input was carefully formatted to ensure clarity and to guide the model in adhering strictly to the given context.

<h4>2- Initial Response Generation:</h4> The model was first tested to determine if additional information was required to generate a comprehensive response. This was facilitated by a preliminary retrieval step where the model decided between "[Retrieval]" and "[No Retrieval]" based on the necessity of additional context.

<h4>3- Retrieval Decision and Response Generation:</h4> If the model determined that additional context was needed ("[Retrieval]"), it would retrieve the necessary information and generate a response grounded in this enriched context. If not ("[No Retrieval]"), the response was generated based on the initial prompt alone.

<h4>4- Scoring and Evaluation:</h4> Responses were evaluated using predefined metrics to determine relevance, support, and utility. Scores were calculated to assess how well the responses matched the provided context and accurately answered the prompts.Cosine similarity and log probability scores were computed to further evaluate the quality of the responses.

- Cosine Similarity Calculation: Cosine similarity is a measure used to determine the similarity between two vectors, in this case, the vectors representing the generated response and the ground truth. It is calculated as the cosine of the angle between the vectors, indicating how closely aligned they are in the vector space. Higher cosine similarity values suggest greater similarity between the responses.

- Log Probability: Log probability scores provide additional insights into the likelihood of the generated response given the context and prompt. Higher log probability scores indicate that the model is more confident in its generated response.

<h4>5- Accuracy Calculation:</h4> The overall performance was summarized by calculating the accuracy of the responses.This was done by comparing the generated answers with the expected ones.

<h4>Example Evaluation Results for the Relationship Between Breath Methane Positivity and Delayed Transit Constipation : </h4>

| Question                                                                                                  | Prediction                                                                                               | Ground Truth                                                                                                                                                       | Cosine Similarity        |
|----------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------|
| What is the relationship between breath methane positivity and delayed transit constipation in patients? | The study found that patients with delayed transit constipation were more likely to be breath methane positive compared to those with normal transit constipation or healthy controls. | Breath methane positivity is more common and higher in patients with objectively proven delayed transit constipation. In fact, the positivity to the lactulose methane breath test (LMBT) was significantly higher in delayed transit patients (58.8%) compared to healthy controls (12.2%) or normal transit patients (13.3%), with delayed transit being the only independent factor for LMBT positivity. | 0.7757343053817749       |

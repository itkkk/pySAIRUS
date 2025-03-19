This folder contains a dict of this form:

{key: (n,m), value: "string"}

Where the key is a couple of ids of two users and value is a string having (pseudo)JSON (therefore, a string) format with this general form:

<json>
{
	"improved_chain_of_thought": ["..."],
	"explanation": ["..."]
	"similarity_score": integer
}
</json>

where improved_chain_of_thought and explanation are strings and similarity_score is an integer.

PLEASE NOTE THAT COULD THERE ARE ITEMS OF DICT THAT NOT HAVING THIS STRUCTURE, BUT THIS IS GENERAL STRUCTURE.

response_dict is union of all other dicts.
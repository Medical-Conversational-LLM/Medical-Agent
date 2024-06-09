"""
This files is clean and fix invalid json structure from csv file ot produce a valid json syntax
"""

class Cleaner:
    pos = 0
    text = ""
    result = ""

    def clean(self, text):
        self.pos = 0
        self.text = text
        self.result = ""

        while self.pos < len(self.text):

            if self.char() == "{":
                self.result += "{"
                self.advance()
                continue
            elif self.char() == ",":
                if self.result_head() == ",":
                    self.advance()
                    continue
                self.result += ","
                self.advance()
                continue
            elif self.char() == "}":
                self.remove_extra_comma()
                self.result += "}"
                self.advance()
                continue
            elif self.char() == "[":
                self.result += "["
                self.advance()
                continue
            elif self.char() == "]":
                self.remove_extra_comma()
                self.result += "]"
                self.advance()
                continue
            elif self.char() == ":":
                self.result += ":"
                self.advance()
                continue
            elif self.char() == "\"":
                self.advance()
                self.result += "\"" + \
                    self.consume_quote("\"")
                continue
            elif self.char() == "'":
                self.advance()
                self.result += "\"" + \
                    self.consume_quote("'", "\"")
                continue

            self.advance()

        return self.result

    def consume_quote(self, end, replace=None):
        result = ""

        char_to_replace = replace if replace is not None else end
        
        while self.pos < len(self.text):
            char = self.char()
            if char == end:
                next = self.peek()

                text_endings = [",", "]", "}", ":"]
                at_end = next in text_endings
                
                if at_end:
                    result += char_to_replace
                    self.advance()
                    break
              
            if char == char_to_replace:
                char = "\\" + char_to_replace
                     

            result += char
            self.advance()

        return result

    def advance(self):
        self.pos += 1
        return self.char()

    def peek(self, offset=1):
        return self.char(self.pos + offset)

    def result_head(self):
        return self.char_of(self.result, len(self.result) - 1)

    def char(self, pos=None):
        return self.char_of(self.text, self.pos if pos == None else pos)

    def remove_extra_comma(self):
        if self.result_head() == ",":
            self.result = self.result.strip(",")

    def char_of(self, text, pos):
        if pos >= len(text) or pos < 0:
            return ""

        char = text[pos]

        if isinstance(text, bytes):
            return chr(char)

        return char


import json

# Function to clean a single row
def clean_row(row):
    # # Remove the leading text from the 'question' field
    # if row['question'].startswith("Here is the converted open-ended question and answer:\n\n**Open-ended Question:**"):
    #     row['question'] = row['question'].replace("Here is the converted open-ended question and answer:\n\n**Open-ended Question:**", "").strip()
    
    # Extract the answer from the 'question' field and move it to a new index
    answer_start = row['question'].find("Answer")
    if (answer_start != -1):
        answer = row['question'][answer_start + len("Answer:")-1:].strip()
        row['question'] = row['question'][:answer_start].strip()
        row['answer'] = answer
        import json
        import re

        # Extract contexts from the 'context' field
        try:
            context_str = row['context'].replace("'", '"').replace('\n', ' ')
            context_str = re.sub(r'\\(?!["\\/bfnrt])', r'\\\\', context_str)  # Escape backslashes
            # context_data = json.loads(context_str)

            context_str = Cleaner().clean(context_str)
            context_data = json.loads(context_str)
            row['contexts'] = context_data.get('contexts', [])
            # print('done',row['contexts'])
        except json.JSONDecodeError as e:
            # row['contexts'] = []
            print(e)


    # Remove the 'long_answer' and 'final_decision' fields
    if 'long_answer' in row:
        del row['long_answer']
    if 'final_decision' in row:
        del row['final_decision']
    if 'context' in row:
        del row['context']

    return row

# Load the JSON file
with open('storage/output/randomized_dataset3.json', 'r') as file:
    data = json.load(file)

    print('test')

# Clean each row in the dataset
cleaned_data = [clean_row(row) for row in data]


# Change 'contexts' column back to 'context' after processing
for row in cleaned_data:
    if 'contexts' in row:
        row['context'] = row.pop('contexts')

# Save the cleaned data to a new JSON file
with open('cleaned_file.json', 'w') as file:
    json.dump(cleaned_data, file, indent=4)

print("Cleaning complete and saved to 'cleaned_file.json'.")

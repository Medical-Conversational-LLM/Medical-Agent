from reader.reader import Reader
from reader.line_to_json import LineToJson
from writer.writer import Writer
from writer.json import JsonWriter
import re

def fix_output_by_matching(matches:list[str], options: list[str], output:str):
    if len(matches) > 0:
        for match in matches:
            if match in options:
                return "[{}]".format(match)
        
    for item in options:
        if item in output:
            return "[{}]".format(item)

def fix_utility(output:str):
    
    output = output.replace("Utility:Utility", "Utility", 1)

    cleaned_text = re.findall(r'Utility:\s*\d', output, re.IGNORECASE)

    if(len(cleaned_text) == 0):
        return None
 
    return "[{}]".format(cleaned_text[0])


def fix_output(line):
    output = line["output"].strip()        
    
    if output.startswith("[") == True and  output.endswith("]") == True :
        if line["task"] == "utility":            
            line['output'] = fix_utility(output)            
        return line 
    
    pattern = r'\[(.*?)\]'
    matches = re.findall(pattern, output)
    
    if line["task"] == "retrieval":
        line['output'] = fix_output_by_matching(matches, ['No Retrieval', 'Retrieval'], output)
    
    if line["task"] == "utility":            
            line['output'] = fix_utility(output)    

    if line['output'].startswith("output:"):
        line['output'] = line['output'].replace("output:", "", 1)

    if line['output'].startswith("Output:"):
        line['output'] = line['output'].replace("Output:", "", 1)

    line['output']=line['output'].strip()
    
    return line

def fix_output_dataset(reader:Reader, writer:Writer ):
    for line in reader.read():        
        line = fix_output(line)
        
        if line["output"] is None:
            continue

        writer.append(line)
            
    writer.close()


if __name__ == "__main__":
    
    writer = JsonWriter("./storage/output/critic_clean.json")    
    writer.truncate()
    
    fix_output_dataset(
        LineToJson("./storage/output/retriver.json"),
        writer
    )
from fix_output_dataset import fix_output_dataset
from reader.line_to_json import LineToJson
from reader.memory import MemoryReader

from writer.json import JsonWriter

def merge_output_dataset(
    dataset_1_file,
    dataset_2_file,
    output_file
):
    reader_1 = LineToJson(dataset_1_file)
    reader_2 = LineToJson(dataset_2_file)
    writer = JsonWriter(output_file)
    final_data = dict()
    
    i = -1
    for line in reader_1.read():
        i+=1
        if(final_data.get(line["input"])):            
            continue
        final_data[line["input"]] = line
        
    print(len(final_data.keys()))
    for line in reader_2.read():
        i+=1
        
        if(final_data.get(line["input"])):            
            continue
        
        final_data[line["input"]] = line
            
    writer.truncate()
    
    fix_output_dataset(
        MemoryReader(final_data.values()),
        writer
    )

merge_output_dataset(
    "./storage/output/critic 4.json",
    "./storage/output/critic.json",
    "./storage/output/critic_merged_clean.json",
)
def compact_summary_helper(line):
    matches = ['Model:', 'Total params:', 'Trainable params:', 'Non-trainable params:']
    if any(x in line for x in matches):
        print("*", line)

def print_model_summary_compact(model):
    model.summary(print_fn=compact_summary_helper)
    # print ("* model name:", model.name)
    print ("* # layers: ", len(model.layers))
    

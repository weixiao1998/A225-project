from data_loader import DataLoader

save_files = ["./data/neg.txt", "./data/pos.txt"]
data_file = "./data/origin/neg-data[850001-851000].txt"

loader = DataLoader("stop_words.txt")
loader.process_original_data(data_file,save_files)

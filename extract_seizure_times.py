def extract_seizure_times(file_name, recording):
    with open(file_name, 'r', encoding='latin-1') as f:
      content = f.read()

    file_info = content.split("\n\n")
    for info in file_info:
        if recording in info:
            seizure_times = []
            lines = info.split("\n")
            for line in lines:
                if "Seizure Start Time" in line:
                    seizure_start = int(line.split(":")[1].strip().split(" ")[0])
                    seizure_times.append([seizure_start])
                elif "Seizure End Time" in line:
                    seizure_end = int(line.split(":")[1].strip().split(" ")[0])
                    seizure_times[-1].append(seizure_end)

            if len(seizure_times) == 0:
                print("No seizures found in the file.")
                return []

            return seizure_times

    print("File not found.")
    return []


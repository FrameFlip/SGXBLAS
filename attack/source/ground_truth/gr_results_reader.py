import os.path
from typing import Dict, Tuple, List
from br_info_reader import read_br_info


GR_EXP_CONF = "cifar10_resnet34"


def count_file_lines(filepath: str, upper_bound: int = -1) -> int:
    n_lines: int = 0
    with open(filepath, "r") as f:
        for line in f:
            n_lines += 1
            if upper_bound != -1 and n_lines > upper_bound:
                break
    return n_lines


def get_acc_line(filepath: str, upper_bound: int = -1) -> str:
    clean_lastline: str = None
    line_cnt: int = 0
    with open(filepath, "r") as f:
        for line in f:
            line_cnt += 1
            if line_cnt == upper_bound:
                clean_lastline = line
                break
            if line.startswith('*') and not line.startswith('**'):
                clean_lastline = line
                break
    return clean_lastline


def get_clean_features() -> Dict:
    clean_filepath: str = f"./results/{GR_EXP_CONF}/clean/log.txt"
    clean_lines: int = count_file_lines(clean_filepath)
    clean_resline: str = get_acc_line(clean_filepath)

    return {'nlines': clean_lines, 'resline': clean_resline}


fault_dict: Dict[int, str] = {
    -1: "unknown",
    0: "warning message",
    1: "checker model accuracy drop",
    2: "task & checker model accuracy drop",
    3: "timeout",
    4: "terminated (Segmentation fault)",
    5: "terminated (Aborted)",
}


# feature line 
def classify_fault(feature_line: str, clean_resline: str) -> int:
    clean_taskacc: str = clean_resline.split("\t")[0] # clean task model accuracy
    fault_type: int = -1
    if feature_line.startswith('**'):
        # e.g. ** On entry to DGEMM  parameter number  8 had an illegal value
        fault_type = 0
    elif feature_line.startswith(clean_taskacc):
        # e.g. Note: *97.8068%	0.4751%
        fault_type = 1
    elif feature_line.startswith('*'):
        # e.g. *5.4711%	4.2676%
        fault_type = 2
    elif feature_line.startswith('timeout'):
        # e.g. timeout: sending signal TERM to command ‘./results/br51/poc.out’
        fault_type = 3
    elif feature_line.startswith('Segmentation'):
        # e.g. Segmentation fault
        fault_type = 4
    elif feature_line.startswith('Aborted'):
        # e.g Aborted
        fault_type = 5
    else: # not listed above? weired ...
        fault_type = -1

    return fault_type


def read_gr_results() -> Dict[int, List[int]]:
    gr_results = {} # maps fault_type (int) to a list of matched br_idxs

    # features of running on a clean ML library
    clean_feats = get_clean_features()

    total_brs: int = read_br_info()
    for br_idx in range(total_brs):
        br_dir = f"./results/{GR_EXP_CONF}/br{br_idx}"

        if not os.path.exists(br_dir):
            print(f"[warning] GR results of branch br{br_idx} not found, skipped")
            continue

        logfile_path: str = os.path.join(br_dir, "log.txt")
        fault_flag: bool = False

        n_lines = count_file_lines(logfile_path, upper_bound=clean_feats['nlines'])
        res_line = get_acc_line(logfile_path, n_lines)

        # (1) check the no. of lines in the log file
        if  n_lines != clean_feats['nlines']:
            fault_flag = True
            # print(f"[check] ./results/br{br_idx} | ")
            # print(f"    Note: {res_line}")
            
        # (2) check the last line of the log file
        # res_line = get_acc_line(logfile_path)
        elif res_line != clean_feats['resline']:
            fault_flag = True
            # print(f"[check] ./results/br{br_idx} | {fault_dict[classify_fault(res_line, clean_feats['resline'])]}")
            # print(f"    Note: {res_line}")
            # continue

        if fault_flag:
            fault_type: int = classify_fault(res_line.strip(), clean_feats['resline'])
            if fault_type not in gr_results:
                gr_results[fault_type] = []
            gr_results[fault_type].append(br_idx)

            print(f"[check] ./results/{GR_EXP_CONF}/br{br_idx} | {fault_dict[fault_type]}")
            print(f"    Note: {res_line.strip()}")

    return gr_results



def main():
    if not os.path.exists("./results"):
        print("GR results not found, terminated")
        return
    
    gr_results: Dict[int, int] = read_gr_results()
    print()
    print(f"{'Fault Type':40}{'Nums':6}{'br_idx Details'}")
    for k in gr_results.keys():
        print(f"{fault_dict[k]:40}{len(gr_results[k]):<6}{gr_results[k]}")

        if k == 0: # warning message, check accuracy drops
            # warning_res maps a possible last-line fault type of a warning-message log, 
            #   to a list of matched br_idxs
            # - clean means the model accuracy is not degraded
            # - checker only means the task model is clean, but the checker is damaged
            # - task & checker means both the task and the checker models are damaged
            # - abnormal means the last line of the log does not show the model accuracies 
            warning_res: Dict[str, List[int]] = {'clean': [], 'checker only': [], 'task & checker': [], 'abnormal': []}
            clean_feats = get_clean_features()
            clean_taskacc: str = clean_feats['resline'].split("\t")[0]

            for wbr_idx in gr_results[k]:
                logfile_path: str = f"./results/{GR_EXP_CONF}/br{wbr_idx}/log.txt"
                
                # Reference: https://stackoverflow.com/questions/46258499/how-to-read-the-last-line-of-a-file-in-python
                with open(logfile_path, 'rb') as f:
                    try:  # catch OSError in case of a one line file 
                        f.seek(-2, os.SEEK_END)
                        while f.read(1) != b'\n':
                            f.seek(-2, os.SEEK_CUR)
                    except OSError:
                        f.seek(0)
                    last_line = f.readline().decode()

                # print(last_line.strip())
                if last_line.startswith(clean_feats['resline']):
                    warning_res['clean'].append(wbr_idx)
                elif last_line.startswith(clean_taskacc):
                    warning_res['checker only'].append(wbr_idx)
                elif last_line.startswith('*'):
                    warning_res['task & checker'].append(wbr_idx)
                else:
                    warning_res['abnormal'].append(wbr_idx)
                
            for wt in warning_res.keys():
                print(f"    {wt:36}{len(warning_res[wt]):<6}{warning_res[wt]}")



if __name__ == "__main__":
    main()

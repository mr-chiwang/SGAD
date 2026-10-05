import numpy as np
import os
import sys
sys.path.append("../")
from loguru import logger
import matplotlib.pyplot as plt

from utils.common import test_dir_if_not_create
from utils.geo import relative_pose_error

"""Matching Accuracy Evaluation ===================================================="""
class MMAEval(object):
    """
    """
    cfg_dft = {
        "root_path": "",
        "folder_name":"",
        "baseline_name": "",
        "challenger_name": "",
        "dataset_name": "",
        "phi_list": [0.25, 0.5, 1.0, 2.0, 5.0],
        "px_list": [1,2,3], # order is px-1
        "ratio_postfix": "ratios.txt",
        "name_postfix": "ratios_name.txt",
        "base_ratio_postfix": "ratios.txt",
        "base_name_postfix": "ratios_name.txt",
        "output_path": "",
    }

    def __init__(self, cfg={}) -> None:
        """
        """
        self.config = {**self.cfg_dft, **cfg}
        self.root_path = self.config["root_path"]
        self.folder_name = self.config["folder_name"]
        self.baseline_name = self.config["baseline_name"]
        self.challenger_name = self.config["challenger_name"]
        self.dataset_name = self.config["dataset_name"]
        self.phi_list = self.config["phi_list"]
        self.px_list = self.config["px_list"]
        self.ratio_postfix = self.config["ratio_postfix"]
        self.name_postfix = self.config["name_postfix"]
        self.output_path = self.config["output_path"]
        test_dir_if_not_create(self.output_path)

    def assemble_paths(self):
        """ assemble paths for each phi
        Returns:
            path_dict = {
                phi: { 
                    baseline_path: baseline_path, 
                    baseline_name_path: baseline_name_path,
                    challenger_path: challenger_path,
                    challenger_name_path: challenger_name_path,
                    }
            }
        """
        self.path_dict = {}

        for phi in self.phi_list:
            baseline_path = os.path.join(self.root_path, self.folder_name, self.baseline_name+f"_"+self.ratio_postfix)
            baseline_name_path = os.path.join(self.root_path, self.folder_name, self.baseline_name+f"_"+self.name_postfix)
            challenger_path = os.path.join(self.root_path, self.folder_name, self.challenger_name+f"_phi_{phi}_"+self.ratio_postfix)
            challenger_name_path = os.path.join(self.root_path, self.folder_name, self.challenger_name+f"_phi_{phi}_"+self.name_postfix)
            self.path_dict[phi] = {
                "baseline_path": baseline_path,
                "baseline_name_path": baseline_name_path,
                "challenger_path": challenger_path,
                "challenger_name_path": challenger_name_path,
            }
        
        return self.path_dict

    def eval_single_phi(self, phi):
        """
        """
        baseline_path = self.path_dict[phi]["baseline_path"]
        baseline_name_path = self.path_dict[phi]["baseline_name_path"]
        challenger_path = self.path_dict[phi]["challenger_path"]
        challenger_name_path = self.path_dict[phi]["challenger_name_path"]

        baseline_auc, challenger_auc = self.get_MMA(baseline_path, baseline_name_path, challenger_path, challenger_name_path, phi)

        return baseline_auc, challenger_auc

    def get_MMA(self, baseline_path, baseline_name_path, challenger_path, challenger_name_path, phi=""):
        """ mma file format:
                each line: ratios@px-1 ratios@px-2 ... ratios@px-10

        """
        baseline_ratios = np.loadtxt(baseline_path)
        baseline_name = self.load_name_txt(baseline_name_path)
        logger.info(f'load baseline {self.baseline_name} ratios shape = {baseline_ratios.shape}')

        challenger_ratios = np.loadtxt(challenger_path)
        challenger_name = self.load_name_txt(challenger_name_path)
        logger.info(f'load challenger {self.challenger_name} ratios shape = {challenger_ratios.shape}')

        # get the values according to the px_list
        baseline_ratios_valid = []
        challenger_ratios_valid = []
        for px in self.px_list:
            ratio_id = px - 1
            assert ratio_id < 10, f"{ratio_id} should be less than 10"
            assert ratio_id >= 0, f"{ratio_id} should be greater than 0"

            baseline_ratio = baseline_ratios[:, ratio_id]
            mean_baseline_ratio = np.mean(baseline_ratio)
            baseline_ratios_valid.append(mean_baseline_ratio)

            challenger_ratio = challenger_ratios[:, ratio_id]
            mean_challenger_ratio = np.mean(challenger_ratio)
            challenger_ratios_valid.append(mean_challenger_ratio)

        logger.success(f"get MMA ratios @ px: {self.px_list}")
        logger.success(f"baseline {self.baseline_name} mean ratios: {baseline_ratios_valid}")
        logger.success(f"challenger {self.challenger_name}_{phi} mean ratios: {challenger_ratios_valid}")

        return baseline_ratios_valid, challenger_ratios_valid
    
    def run(self):
        """
        """
        self.assemble_paths()
        for phi in self.phi_list:
            self.eval_single_phi(phi)

    def load_name_txt(self, name_file): 
        """
        """
        rt_names = []
        with open(name_file, "r") as f:
            names = f.readlines()
            for name in names:
                if name=="":continue
                name = name.replace('\n','')
                rt_names.append(name)
        return rt_names


"""Pose Estimation Evaluation ======================================================"""

class PoseAUCEval(object):
    """
        TODO: add the sample function
    """

    cfg_dft = {
        "root_path": "",
        "folder_name":"",
        "baseline_name": "",
        "challenger_name": "",
        "dataset_name": "",
        "phi_list": [0.25, 0.5, 1.0, 2.0, 5.0],
        "err_postfix": "pose_errs.txt",
        "name_postfix": "pose_err_names.txt",
        "base_err_postfix": "pose_errs.txt",
        "base_name_postfix": "pose_err_names.txt",
        "output_path": "",
        "fix_len": None,
    }

    def __init__(self, cfg={}) -> None:
        """
        """
        self.config = {**self.cfg_dft, **cfg}
        self.root_path = self.config["root_path"]
        self.folder_name = self.config["folder_name"]
        self.baseline_name = self.config["baseline_name"]
        self.challenger_name = self.config["challenger_name"]
        self.dataset_name = self.config["dataset_name"]
        self.phi_list = self.config["phi_list"]
        self.err_postfix = self.config["err_postfix"]
        self.name_postfix = self.config["name_postfix"]
        self.base_err_postfix = self.config["base_err_postfix"]
        self.base_name_postfix = self.config["base_name_postfix"]
        self.output_path = self.config["output_path"]
        if self.config["fix_len"] is not None:
            self.fix_len = self.config["fix_len"]
        else:
            self.fix_len = None
        test_dir_if_not_create(self.output_path)        

    def assemble_paths(self):
        """ assemble paths for each phi
        Returns:
            path_dict = {
                phi: { 
                    baseline_path: baseline_path, 
                    baseline_name_path: baseline_name_path,
                    challenger_path: challenger_path,
                    challenger_name_path: challenger_name_path,
                    }
            }
        """
        self.path_dict = {}

        for phi in self.phi_list:
            baseline_path = os.path.join(self.root_path, self.folder_name, self.baseline_name+"_"+self.err_postfix)
            baseline_name_path = os.path.join(self.root_path, self.folder_name, self.baseline_name+"_"+self.name_postfix)
            challenger_path = os.path.join(self.root_path, self.folder_name, f'{self.challenger_name}-{phi}_{self.err_postfix}')
            challenger_name_path = os.path.join(self.root_path, self.folder_name, f'{self.challenger_name}-{phi}_{self.name_postfix}')
            self.path_dict[phi] = {
                "baseline_path": baseline_path,
                "baseline_name_path": baseline_name_path,
                "challenger_path": challenger_path,
                "challenger_name_path": challenger_name_path,
            }
        
        return self.path_dict
    
    def eval_single_phi(self, phi):
        """
        """
        baseline_path = self.path_dict[phi]["baseline_path"]
        baseline_name_path = self.path_dict[phi]["baseline_name_path"]
        challenger_path = self.path_dict[phi]["challenger_path"]
        challenger_name_path = self.path_dict[phi]["challenger_name_path"]

        baseline_auc, challenger_auc = self.get_PoseAUC(baseline_path, baseline_name_path, challenger_path, challenger_name_path, phi)

        return baseline_auc, challenger_auc

    def get_PoseAUC(self, baseline_path, baseline_name_path, challenger_path, challenger_name_path, phi=""):
        """
        """

        logger.success(f"load baseline {self.baseline_name} pose errors from {baseline_path}")
        logger.success(f"load challenger {self.challenger_name}_{phi} pose errors from {challenger_path}")

        try:
            baseline_errs = np.loadtxt(baseline_path)
        except ValueError as e:
            logger.critical(f"{baseline_path} load error")
            
        baseline_name = self.load_name_txt(baseline_name_path)
        # get the idx of nans
        idx_nans_baseline = np.argwhere(np.isnan(baseline_errs))
        baseline_errs[idx_nans_baseline] = [180, 180]

        # get the repeated idx of baseline_name
        idx_repeated = []
        for i in range(len(baseline_name)):
            if baseline_name[i] in baseline_name[:i]:
                idx_repeated.append(i)
        
        # delete the repeated idx
        baseline_errs = np.delete(baseline_errs, idx_repeated, axis=0)
        baseline_name = np.delete(baseline_name, idx_repeated, axis=0)
        
        logger.info(f'load baseline {self.baseline_name} pose errors shape = {baseline_errs.shape}')

        try:
            challenger_errs = np.loadtxt(challenger_path)
        except Exception as e:
            logger.critical(f"{challenger_path} load error")
        challenger_name = self.load_name_txt(challenger_name_path)
        # get the idx of nans
        idx_nans_challenger = np.argwhere(np.isnan(challenger_errs))
        challenger_errs[idx_nans_challenger] = [180, 180]

        # get the repeated idx of challenger_name
        idx_repeated = []
        for i in range(len(challenger_name)):
            if challenger_name[i] in challenger_name[:i]:
                idx_repeated.append(i)
        
        # delete the repeated idx
        challenger_errs = np.delete(challenger_errs, idx_repeated, axis=0)
        challenger_name = np.delete(challenger_name, idx_repeated, axis=0)

        logger.info(f'load challenger {self.challenger_name} pose errors shape = {challenger_errs.shape}')

        # get the common idx of baseline_name and challenger_name
        common_name = np.intersect1d(baseline_name, challenger_name)
        logger.info(f"common name shape {common_name.shape}")

        # get the common idx of baseline_name and challenger_name
        common_idx_baseline = []
        common_idx_challenger = []
        for i in range(len(common_name)):
            common_idx_baseline.append(np.argwhere(baseline_name == common_name[i]))
            common_idx_challenger.append(np.argwhere(challenger_name == common_name[i]))

        common_idx_baseline = np.squeeze(np.array(common_idx_baseline))
        common_idx_challenger = np.squeeze(np.array(common_idx_challenger))

        logger.info(f"common idx baseline shape {common_idx_baseline.shape}")

        # get the common pose errs
        baseline_errs = baseline_errs[common_idx_baseline]
        challenger_errs = challenger_errs[common_idx_challenger]

        # get the fix len
        if self.fix_len is not None:
            baseline_errs = baseline_errs[:self.fix_len]
            challenger_errs = challenger_errs[:self.fix_len]

        logger.info(f"common baseline pose errs shape {baseline_errs.shape}")
        logger.info(f"common challenger pose errs shape {challenger_errs.shape}")

        from utils.geo import aggregate_pose_auc_simp

        # baseline 
        baseline_err_mean = np.mean(baseline_errs, axis=0)
        baseline_auc = aggregate_pose_auc_simp(baseline_errs, self.dataset_name)
        logger.success(f"baseline {self.baseline_name} pose auc: {baseline_auc}")

        # challenger
        challenger_err_mean = np.mean(challenger_errs, axis=0)
        challenger_auc = aggregate_pose_auc_simp(challenger_errs, self.dataset_name)
        logger.success(f"challenger {self.challenger_name}_{phi} pose auc: {challenger_auc}")

        logger.success(f"baseline error mean: {baseline_err_mean}")
        logger.success(f"challenger error mean: {challenger_err_mean}")

        return baseline_auc, challenger_auc

    def run(self):
        """
        """
        best_phi = -1
        best_auc = -1
        
        self.assemble_paths()
        for phi in self.phi_list:
            baseline_auc, challenge_auc = self.eval_single_phi(phi)

            total_diff = 0
            for k in baseline_auc.keys():
                temp_diff = - baseline_auc[k] + challenge_auc[k]
                total_diff += temp_diff
            
            total_diff = total_diff / len(baseline_auc.keys())
            
            if total_diff > best_auc:
                best_auc = total_diff
                best_phi = phi
        
        logger.critical(f"best phi: {best_phi}, best auc diff: {best_auc}")
            

    def run_without_phi(self):
        """
        """
        baseline_path = os.path.join(self.root_path, self.folder_name, self.baseline_name+"_"+self.base_err_postfix)
        baseline_name_path = os.path.join(self.root_path, self.folder_name, self.baseline_name+"_"+self.base_name_postfix)
        challenger_path = os.path.join(self.root_path, self.folder_name, self.challenger_name+"_"+self.err_postfix)
        challenger_name_path = os.path.join(self.root_path, self.folder_name, self.challenger_name+"_"+self.name_postfix)

        self.get_PoseAUC(baseline_path, baseline_name_path, challenger_path, challenger_name_path)
    
    
    def load_name_txt(self, name_file): 
        """
        """
        rt_names = []
        with open(name_file, "r") as f:
            names = f.readlines()
            for name in names:
                if name=="":continue
                name = name.replace('\n','')
                rt_names.append(name)
        return rt_names

        
"""Area Matching Evaluation ======================================================"""

class AMEval(object):
    """Evaluate region matching with ACR, IoU, and AOR metrics."""

    AM_default_configs = {
        "root_path": "",
        "name": "",
        "res_folder": "ratios",
        "AOR_post": "_aor.txt",
        "ACR_post": "_acr.txt",
        "IOU_post": "_iou.txt",
        "file_name_post": "_ameval_names.txt",
        "time_post": "_area_match_time.txt",
        "aor_thresholds": [0.3, 0.5, 0.7],
        "iou_thresholds": [0.1, 0.3, 0.5],
        "instance_aor_acc_post": "_instance_aor_acc.txt",
        "instance_iou_acc_post": "_instance_iou_acc.txt",
    }

    def __init__(self, configs={}) -> None:
        self.configs = {**self.AM_default_configs, **configs}
        logger.info(f"Initialize AM Evaluation with configs: {self.configs}")
        self.root_path = self.configs["root_path"]
        self.name = self.configs["name"]
        self.res_folder = self.configs["res_folder"]
        self.AOR_post = self.configs["AOR_post"]
        self.ACR_post = self.configs["ACR_post"]
        self.IOU_post = self.configs["IOU_post"]
        self.file_name_post = self.configs["file_name_post"]
        self.time_post = self.configs["time_post"]
        self.aor_thresholds = self.configs["aor_thresholds"]
        self.iou_thresholds = self.configs["iou_thresholds"]
        self.instance_aor_acc_post = self.configs["instance_aor_acc_post"]
        self.instance_iou_acc_post = self.configs["instance_iou_acc_post"]
        self.output_path = os.path.join(self.root_path, "AMEval")
        self._test_and_create_dir(self.output_path)

        # Metric arrays are loaded lazily from per-instance result files.
        self.acrs = None
        self.ious = None
        self.aors = None
        self.times = None
        self.names = None

    @staticmethod
    def _test_and_create_dir(path):
        if not os.path.exists(path):
            os.makedirs(path)

    def load_acr_file(self):
        """Load one ACR value per line."""
        acr_file = os.path.join(self.root_path, self.res_folder, self.name + self.ACR_post)
        try:
            self.acrs = np.loadtxt(acr_file)
            logger.info(f"Loaded ACRs with shape: {self.acrs.shape}")
        except Exception as e:
            logger.error(f"Failed to load ACR file: {e}")
            raise
        return self.acrs

    def load_iou_file(self):
        """Load comma-separated IoU values for each instance."""
        iou_file = os.path.join(self.root_path, self.res_folder, self.name + self.IOU_post)
        self.ious = []
        try:
            with open(iou_file, 'r') as f:
                for line in f:
                    values = line.strip().split(',')
                    instance_ious = [float(v) for v in values]
                    self.ious.append(instance_ious)
            logger.info(f"Loaded IoUs for {len(self.ious)} instances")
        except Exception as e:
            logger.error(f"Failed to load IoU file: {e}")
            raise
        return self.ious

    def load_aor_file(self):
        """Load comma-separated AOR values for each instance."""
        aor_file = os.path.join(self.root_path, self.res_folder, self.name + self.AOR_post)
        self.aors = []
        try:
            with open(aor_file, 'r') as f:
                for line in f:
                    values = line.strip().split(',')
                    instance_aors = [float(v) for v in values]
                    self.aors.append(instance_aors)
            logger.info(f"Loaded AORs for {len(self.aors)} instances")
        except Exception as e:
            logger.error(f"Failed to load AOR file: {e}")
            raise
        return self.aors

    def load_time_file(self):
        """Load recorded matching times."""
        time_file = os.path.join(self.root_path, self.res_folder, self.name + self.time_post)
        try:
            self.times = np.loadtxt(time_file)
            logger.info(f"Loaded times with shape: {self.times.shape}")
        except Exception as e:
            logger.error(f"Failed to load time file: {e}")
            raise
        return self.times

    def load_file_names(self):
        """Load evaluated instance names."""
        name_file = os.path.join(self.root_path, self.res_folder, self.name + self.file_name_post)
        try:
            self.names = np.loadtxt(name_file, dtype=str)
            logger.info(f"Loaded {len(self.names)} instance names")
        except Exception as e:
            logger.error(f"Failed to load names: {e}")
            raise
        return self.names

    def calc_acr(self):
        """Compute mean ACR across all instances."""
        if self.acrs is None:
            self.load_acr_file()
        mean_acr = np.mean(self.acrs)
        logger.success(f"Mean ACR: {mean_acr:.4f}")
        return mean_acr

    def _calc_instance_accuracy(self, instance_values, thresholds):
        """Compute one instance's accuracy at each threshold."""
        accuracies = {}
        total = len(instance_values)
        if total == 0:
            return {thresh: 0.0 for thresh in thresholds}
        
        for thresh in thresholds:
            correct = sum(1 for v in instance_values if v >= thresh)
            acc = correct / total if total > 0 else 0.0
            accuracies[thresh] = acc
        return accuracies

    def _calc_accuracy(self, values_list, thresholds, metric_name):
        """Compute per-instance and mean accuracy at each threshold."""
        all_accuracies = {thresh: [] for thresh in thresholds}
        instance_accuracies = []

        # Return zero mean accuracy when no instances are available.
        if not values_list:
            logger.warning(f"No instances found for {metric_name} calculation!")
            return {thresh: 0.0 for thresh in thresholds}, []
        
        for instance_idx, instance_values in enumerate(values_list):
            instance_accs = self._calc_instance_accuracy(instance_values, thresholds)
            instance_accuracies.append(instance_accs)
            for thresh in thresholds:
                all_accuracies[thresh].append(instance_accs[thresh])

        # Average per-instance accuracies independently for each threshold.
        mean_accuracies = {}
        for thresh in thresholds:
            mean_acc = np.mean(all_accuracies[thresh])
            mean_accuracies[thresh] = mean_acc
            logger.success(f"{metric_name} Mean Accuracy @ {thresh}: {mean_acc:.4f}")
        
        return mean_accuracies, instance_accuracies

    def calc_iou_accuracy(self):
        """Compute per-instance and mean IoU accuracy."""
        if self.ious is None:
            self.load_iou_file()
        return self._calc_accuracy(self.ious, self.iou_thresholds, metric_name="IoU")

    def calc_aor_accuracy(self):
        """Compute per-instance and mean AOR accuracy."""
        if self.aors is None:
            self.load_aor_file()
        return self._calc_accuracy(self.aors, self.aor_thresholds, metric_name="AOR")
        
    def save_instance_accuracies(self, instance_accs, thresholds, metric_name):
        """Write per-instance threshold accuracies to the AMEval directory."""
        if metric_name == "IoU":
            output_file = os.path.join(self.output_path, self.name + self.instance_iou_acc_post)
        elif metric_name == "AOR":
            output_file = os.path.join(self.output_path, self.name + self.instance_aor_acc_post)
        else:
            logger.error(f"Unknown metric name: {metric_name}")
            return
        
        # Fall back to generated indices when instance names are unavailable.
        if self.names is None:
            try:
                self.load_file_names()
            except Exception as e:
                logger.warning(f"Failed to load instance names: {e}. Using indices instead.")
                self.names = [f"instance_{i}" for i in range(len(instance_accs))]
        
        # Align the number of names with the number of accuracy records.
        if len(self.names) != len(instance_accs):
            logger.warning(f"Number of instance names ({len(self.names)}) does not match number of instance accuracies ({len(instance_accs)})")
            if len(self.names) < len(instance_accs):
                self.names = list(self.names) + [f"instance_{i+len(self.names)}" for i in range(len(instance_accs) - len(self.names))]
            else:
                self.names = self.names[:len(instance_accs)]
        
        try:
            with open(output_file, 'w') as f:
                # Write a CSV-style header followed by one row per instance.
                header = "instance_name," + ",".join([f"{metric_name}_{thresh}" for thresh in thresholds])
                f.write(header + "\n")

                for i, instance_acc in enumerate(instance_accs):
                    instance_name = self.names[i] if i < len(self.names) else f"instance_{i}"
                    values = [f"{instance_acc[thresh]:.4f}" for thresh in thresholds]
                    line = instance_name + "," + ",".join(values)
                    f.write(line + "\n")
            
            logger.success(f"Saved {len(instance_accs)} instance {metric_name} accuracies to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save instance accuracies: {e}")
            raise

    def run(self):
        """Load region metrics, compute accuracies, and write summaries."""
        logger.info("Starting evaluation...")
        # Load all inputs required by the aggregate metrics.
        self.load_acr_file()
        self.load_iou_file()
        self.load_aor_file()
        try:
            self.load_file_names()
        except Exception as e:
            logger.warning(f"Failed to load instance names: {e}. Using indices instead.")
            
        self.calc_acr()

        # Compute and persist thresholded IoU accuracy.
        iou_accuracies, iou_instance_accs = self.calc_iou_accuracy()
        self.save_instance_accuracies(iou_instance_accs, self.iou_thresholds, "IoU")
        
        # Compute and persist thresholded AOR accuracy.
        aor_accuracies, aor_instance_accs = self.calc_aor_accuracy()
        self.save_instance_accuracies(aor_instance_accs, self.aor_thresholds, "AOR")
        
        logger.info("Evaluation completed.")
        
"""Time Evaluation ======================================================"""

class TimeEval(object):
    """ save as time_name.time_post -> read time_name and report
    """
    time_eval_default_configs = {
        "root_path": "",
        "res_folder": "MMARatios",
        "time_post": "_time.txt",
    }

    def __init__(self, configs={}) -> None:
        """
        """
        self.configs = self.time_eval_default_configs
        self.configs = {**self.configs, **configs}
        logger.info(f"Initialize the Time Evaluation with configs: {self.configs}")
        self.root_path = self.configs["root_path"]
        self.res_folder = self.configs["res_folder"]
        self.time_post = self.configs["time_post"]

        self.time_dict = None

    def load_time_file(self):
        """ load all file with time_post in root_path/res_folder
            1. get all files whose name post is self.time_post
            2. load the time file (record name) and calc the mean time
            3. save the time in self.time_dict
        """
        import glob
        time_files = glob.glob(os.path.join(self.root_path, self.res_folder, "*" + self.time_post))
        logger.info(f"Get {len(time_files)} time files")
        time_dict = {}
        for time_file in time_files:
            name = os.path.basename(time_file).split(".")[0]
            time = np.loadtxt(time_file)
            # calc the mean time
            time = np.mean(time)
            time_dict[name] = time
            logger.info(f"Get {name} mean time: {time}")
        self.time_dict = time_dict
        return time_dict

    def run(self):
        """
        """
        self.load_time_file()

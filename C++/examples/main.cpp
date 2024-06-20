#include "SESync/SESync.h"
#include "SESync/SESync_utils.h"

#include <fstream>

#ifdef GPERFTOOLS
#include <gperftools/profiler.h>
#endif

using namespace std;
using namespace SESync;

bool write_poses = true;
ifstream& operator>>(ifstream& is, Eigen::Matrix<Scalar, -1, -1>& m)
{
    double value;
    is >> value;
    Eigen::CommaInitializer< Eigen::Matrix<Scalar, -1, -1>> init(m, value);
    while (is>> value)
    {
        init, value;  //Eigen::don;t touch code
    }
    init.finished();
    return is;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    cout << "Usage: " << argv[0] << " [input .g2o file]" << endl;
    exit(1);
  }

  size_t num_poses;
  measurements_t measurements = read_g2o_file(argv[1], num_poses);
  cout << "Loaded " << measurements.size() << " measurements between "
       << num_poses << " poses from file " << argv[1] << endl
       << endl;
  if (measurements.size() == 0) {
    cout << "Error: No measurements were read!"
         << " Are you sure the file exists?" << endl;
    exit(1);
  }

  SESyncOpts opts;
  opts.verbose = true; // Print output to stdout

  // Initialization method
  // Options are:  Chordal, Random
  opts.initialization = Initialization::Chordal;

  // Specific form of the synchronization problem to solve
  // Options are: Simplified, Explicit, SOSync
  opts.formulation = Formulation::Simplified;

  // Initial
  opts.num_threads = 4;

#ifdef GPERFTOOLS
  ProfilerStart("SE-Sync.prof");
#endif

  /// RUN SE-SYNC!
  SESyncResult results = SESync::SESync(measurements, opts);

#ifdef GPERFTOOLS
  ProfilerStop();
#endif

  if (write_poses) {
    // Write output
#ifdef WITH_SUITESPARSE
    string filename = "poses_ss.txt";
#else
    string filename = "poses.txt";
#endif
    cout << "Saving final poses to file: " << filename << endl;
    ofstream poses_file(filename);
    poses_file << results.xhat;
    poses_file.close();


    Eigen::Matrix<Scalar,-1,-1> gt_xhat = Eigen::Matrix<Scalar, -1, -1>::Zero(results.xhat.rows(), results.xhat.cols());
    {
        string filename_gt = R"(C:\workspace\git\eiva-sesync\build\bin\RelWithDebInfo\poses_ss.txt)";
       
        ifstream poses_gt_file(filename_gt);
        if (poses_gt_file.is_open()) std::cout << "Success loading gt files"<< std::endl;
        poses_gt_file >> gt_xhat;
    }

    Eigen::MatrixXd err = results.xhat - gt_xhat;
    std::cout<<"Mean err:: " << err.transpose().rowwise().mean();
  }
}

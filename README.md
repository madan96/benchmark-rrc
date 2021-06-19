RRC 2021
=============================================================

[Old README](OldREADME.md)
### Running the code in simulation

To run the code locally, first install [Singularity](https://sylabs.io/guides/3.5/user-guide/quick_start.html) and download the singularity image for RRC 2021

```
singularity pull library://felix.widmaier/trifinger/user:latest
```

A couple extra dependencies are required to run our code. To create the required singularity image, run:
```singularity build --fakeroot image.sif image.def```

Use the `run_locally.sh` script to build the catkin workspace and run commands
inside the singularity image.

to run the Motion Planning with Planned Grasp (MP-PG) on a random goal trajectory, use the following
command:
```bash
./run_locally.sh /path/to/singularity/image.sif ros2 run rrc run_local_episode_traj.py 3 mp-pg
```
Run Cartesian Position Control with Triangulated Grasp (CPC-TG) using:
```
./run_locally.sh /path/to/singularity/image.sif ros2 run rrc run_local_episode_traj.py 3 cpc-tg
```

To evaluate the method, modify the `evaluate_policy.py` script by changing the method you want to evaluate. Currently, you need to change Line#63. Then run:
```
python3 scripts/rrc_evaluate_prestage.py \
    --singularity-image image.sif \
    --package path/to/your/package \
    --output-dir path/to/output_dir
```

Just use `.` for package path if you are running eval from inside the package. Refer to [this page](https://people.tuebingen.mpg.de/felixwidmaier/rrc2021/simulation_phase/evaluation.html) for more details, and [this page](https://people.tuebingen.mpg.de/felixwidmaier/rrc2021/simulation_phase/index.html) for details on submitting the results.
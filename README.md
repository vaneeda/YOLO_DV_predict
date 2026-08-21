# YOLO_DV_predict

YOLO-based fish/plankton detection over DeepVision station zip files.

## Docker: base image + per-variant images

The Docker setup is split in two:

- **`Dockerfile.base`** - the base image: apt packages, Python deps, and the
  CUDA-specific `torch`/`torchvision` reinstall (see `TORCH_CUDA_INDEX` build
  arg). This is the only part that needs internet access, and the only part
  that should ever need to touch it - build it once per host, tagged by which
  CUDA index it was built with (e.g. `yolo_dv_base:cu126`).
- **`Dockerfile-YOLO_DV_predict`** - a thin image built `FROM` that local base
  tag. It only `COPY`s the application code plus one variant's `config.yaml` +
  `best.pt` (via `COPY`, not `git clone` or a bind mount). Because its `FROM`
  resolves to an image already sitting in the local Docker image store, this
  build never touches the network - not even if Docker's build cache gets
  invalidated - so it works on hosts with **no internet access at all**, and a
  built image needs no network access at `docker run` time either.

Each variant's config and weights live under `variants/<name>/` (e.g.
`variants/redfish/`). Weights (`*.pt`) are `.gitignore`d; only `config.yaml`
is tracked.

### Procedure

Assumes the host machine has internet access to build the initial base image
(steps 1-2). Everything after that (steps 3+) works with or without internet.

1. **Clone the repo** on the host machine:
   ```bash
   git clone https://github.com/vaneeda/YOLO_DV_predict.git
   ```

2. **Build the base image once**, tagged by the CUDA index this host's NVIDIA
   driver supports (check with `nvidia-smi` and pick the matching
   `download.pytorch.org/whl/cuXXX` index - it's the *running* host's driver
   that matters, not the machine you happened to clone on if they differ):
   ```bash
   docker build -t yolo_dv_base:cu126 -f Dockerfile.base .
   # older driver, max CUDA 12.5:
   # docker build -t yolo_dv_base:cu124 -f Dockerfile.base --build-arg TORCH_CUDA_INDEX=https://download.pytorch.org/whl/cu124 .
   ```
   Only rebuild this when system packages, Python deps, or the CUDA index
   need to change.

3. **Add the variant's config + weights** under `variants/<name>/`, e.g.:
   ```
   variants/redfish/config.yaml
   variants/redfish/best.pt
   ```

4. **Build the variant image**, pointing at the matching local base tag:
   ```bash
   docker build -t yolo_dv:redfish -f Dockerfile-YOLO_DV_predict \
     --build-arg BASE_IMAGE=yolo_dv_base:cu126 --build-arg VARIANT=redfish .
   ```
   Do this for each variant, and again whenever `predict_DV.py`/`csv2xml.py`/
   `utils.py` or a variant's config/weights change. No internet required -
   this only needs the base image already sitting on this host from step 2.

5. **Transfer the image**, if the machine you'll actually run on is different
   from the one you built on:
   ```bash
   docker save yolo_dv:redfish -o yolo_dv_redfish.tar
   # copy yolo_dv_redfish.tar over however that machine gets files (scp, USB, ...)
   docker load -i yolo_dv_redfish.tar   # on the deployment machine
   ```
   Skip this step if building and running on the same machine.

6. **Run** on the deployment machine:
   ```bash
   docker run -it --gpus "device=1" --rm --user <uid>:<gid> \
     -v /path/to/station/data:/data \
     -v /path/to/output:/temp \
     yolo_dv:redfish
   ```
   - `--gpus "device=N"` selects a specific GPU; the code inside always sees
     it as device `0` regardless of its physical index on the host.
   - `--user <uid>:<gid>` matters on shared/NFS storage with root squash -
     writes as root can silently fail there even though the container itself
     runs fine, since the containers's root gets mapped to an unprivileged
     user by the filesystem.
   - If this deployment host turns out to need a different CUDA index than
     the base image was built with, go back to step 2 on a machine with
     internet access - the variant image itself (step 4) can't fix a
     mismatched `torch` build, only a new base image can.

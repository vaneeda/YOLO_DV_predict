# YOLO_DV_predict

YOLO-based fish/plankton detection over DeepVision station zip files.

## Docker: building per-variant, self-contained images

`Dockerfile-YOLO_DV_predict` bakes the application code plus one variant's
`config.yaml` + `best.pt` directly into the image (via `COPY`, not `git clone`
or a bind mount), so a built image needs no network access at all at
`docker run` time - useful for deployment machines with limited internet
access.

Each variant's config and weights live under `variants/<name>/` (e.g.
`variants/redfish/`). Weights (`*.pt`) are `.gitignore`d; only `config.yaml`
is tracked.

### Procedure

1. **Clone the repo** on a machine with internet access (the build machine -
   this does not need to be the machine you'll eventually run the container
   on):
   ```bash
   git clone https://github.com/vaneeda/YOLO_DV_predict.git
   ```

2. **Add the variant's config + weights** under `variants/<name>/`, e.g.:
   ```
   variants/redfish/config.yaml
   variants/redfish/best.pt
   ```

3. **Build**, tagged per variant, from the repo root:
   ```bash
   docker build -t yolo_dv:redfish -f Dockerfile-YOLO_DV_predict --build-arg VARIANT=redfish .
   ```
   The base image, apt packages, and Python dependencies (including the
   CUDA-specific `torch`/`torchvision` reinstall - see `TORCH_CUDA_INDEX`
   build arg in the Dockerfile) are shared, cached layers across all variant
   builds; only the final `COPY` layers differ per variant.

4. **Transfer the image**, if the deployment machine is different from the
   build machine and has limited/no internet access - it won't be able to
   `docker pull` anything:
   ```bash
   docker save yolo_dv:redfish -o yolo_dv_redfish.tar
   # copy yolo_dv_redfish.tar over however that machine gets files (scp, USB, ...)
   docker load -i yolo_dv_redfish.tar   # on the deployment machine
   ```
   Skip this step if building and running on the same machine.

5. **Run** on the deployment machine:
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
   - `TORCH_CUDA_INDEX` (build arg, step 3) must match what the *deployment*
     host's NVIDIA driver supports, not the build machine's - check with
     `nvidia-smi` on the deployment host and pick the matching
     `download.pytorch.org/whl/cuXXX` index.

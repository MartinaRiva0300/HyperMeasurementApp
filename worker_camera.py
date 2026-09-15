from __future__ import annotations

import multiprocessing as mp
from multiprocessing import shared_memory
import queue
import time

import numpy as np
from loguru import logger

from camera.camera_interface import CameraStatus
from camera.factory import create_camera


def publish_status(frame_queue: mp.Queue, status: CameraStatus) -> None:
    # Status (incl. temperature) must get through even while streaming -- the
    # frame queue is usually full, so drop the oldest packet to make room.
    if frame_queue.full():
        try:
            frame_queue.get_nowait()
        except queue.Empty:
            pass
    frame_queue.put({
        "type": "status",
        "status": {
            "connected": status.connected,
            "acquiring": status.acquiring,
            "backend": status.backend,
            "message": status.message,
            "width": status.width,
            "height": status.height,
            "serial_number": status.serial_number,
            "average_count": status.average_count,
            "exposure_ms": status.exposure_ms,
            "exposure_min_ms": status.exposure_min_ms,
            "exposure_max_ms": status.exposure_max_ms,
            "binning": status.binning,
            "binning_options": list(status.binning_options),
            "gain_db": status.gain_db,
            "offset_x": status.offset_x,
            "offset_y": status.offset_y,
            "pixel_format": status.pixel_format,
            "adc_bit_depth": status.adc_bit_depth,
            "frame_rate_hz": status.frame_rate_hz,
            "reverse_x": status.reverse_x,
            "reverse_y": status.reverse_y,
            "exposure_auto": status.exposure_auto,
            "gain_auto": status.gain_auto,
            "frame_counter": status.frame_counter,
            "raw_peak_count": status.raw_peak_count,
            "board_temp_c": status.board_temp_c,
            "fpa_temp_k": status.fpa_temp_k,
        },
    })


def publish_frame(frame_queue, frame, shared_frame=None) -> None:
    # The frame is passed through exactly as read from the sensor -- no software
    # flip. If the image needs mirroring, use the camera's own ReverseX/ReverseY.
    # ascontiguousarray so the shared-memory write and the embedded ndarray
    # (pickled) path both get a plain C-contiguous array.
    frame = np.ascontiguousarray(frame)
    frame_height, frame_width = frame.shape
    if shared_frame is not None:
        if frame_height > shared_frame.shape[0] or frame_width > shared_frame.shape[1]:
            shared_frame = None
        else:
            shared_frame[:frame_height, :frame_width] = frame

    if frame_queue.full():
        try:
            frame_queue.get_nowait()
        except queue.Empty:
            pass

    packet = {
        "type": "frame",
        "shape": (frame_height, frame_width),
    }
    if shared_frame is None:
        packet["frame"] = frame
    else:
        packet["shared"] = True
    frame_queue.put(packet)


def camera_worker(frame_queue: mp.Queue, control_queue: mp.Queue, camera_config: dict) -> None:
    logger.info("Camera worker starting")
    mode = camera_config.get("mode", "forge")
    target_fps = float(camera_config.get("target_fps", 30.0))
    frame_interval = 1.0 / max(target_fps, 1.0)
    camera = None
    shared_frame = None
    shared_memory_handle = None

    shared_frame_name = camera_config.get("shared_frame_name")
    shared_frame_shape = tuple(camera_config.get("shared_frame_shape", (0, 0)))
    if shared_frame_name and len(shared_frame_shape) == 2:
        shared_memory_handle = shared_memory.SharedMemory(name=shared_frame_name)
        shared_frame = np.ndarray(shared_frame_shape, dtype=np.uint16, buffer=shared_memory_handle.buf)

    def connect_camera(selected_mode: str) -> None:
        nonlocal camera, mode
        mode = selected_mode
        if camera is not None:
            camera.stop_acquisition()
            camera.disconnect()
        camera = create_camera(mode)
        status = camera.connect()
        status.requested_mode = mode
        publish_status(frame_queue, status)
        if status.connected:
            camera.start_acquisition()
            # Read the temperature ONCE on connect; afterwards it is only refreshed
            # on the Poll button (no periodic polling of the serial-over-GigE link).
            if hasattr(camera, "refresh_temperatures"):
                try:
                    camera.refresh_temperatures()
                except Exception:  # noqa: BLE001
                    pass
            publish_status(frame_queue, camera.get_status())
        else:
            logger.warning(f"Camera unavailable: {status.message}")

    connect_camera(mode)

    running = True
    # Whether we WANT to be streaming (intent). Stays True across a link drop so
    # auto-reconnect keeps retrying even after status.acquiring flips to False.
    should_stream = camera is not None and camera.status.acquiring
    last_status_push = 0.0
    last_frame_time = time.time()
    last_reconnect_try = 0.0
    RECONNECT_AFTER_S = 4.0       # no frames this long while "acquiring" = link lost
    RECONNECT_THROTTLE_S = 6.0    # don't hammer reconnect attempts
    try:
        while running:
            try:
                command = control_queue.get_nowait()
                cmd_type = command.get("type")
                if cmd_type == "stop":
                    running = False
                    continue
                if cmd_type == "set_exposure":
                    camera.set_exposure(float(command.get("value", 2.0)))
                    publish_status(frame_queue, camera.get_status())
                    continue
                if cmd_type == "set_average":
                    if hasattr(camera, "set_average"):
                        camera.set_average(int(command.get("value", 1)))
                    publish_status(frame_queue, camera.get_status())
                    continue
                if cmd_type == "set_binning":
                    if hasattr(camera, "set_binning"):
                        camera.set_binning(int(command.get("value", 1)))
                    publish_status(frame_queue, camera.get_status())
                    continue
                if cmd_type == "set_roi":
                    if hasattr(camera, "set_roi"):
                        camera.set_roi(int(command.get("row0", 0)),
                                       int(command.get("row1", 0)),
                                       int(command.get("col0", 0)),
                                       int(command.get("col1", 0)))
                    publish_status(frame_queue, camera.get_status())
                    continue
                if cmd_type == "reset_roi":
                    if hasattr(camera, "reset_roi"):
                        camera.reset_roi()
                    publish_status(frame_queue, camera.get_status())
                    continue
                if cmd_type == "set_option":
                    if hasattr(camera, "set_option"):
                        camera.set_option(command.get("name"), command.get("value"))
                    publish_status(frame_queue, camera.get_status())
                    continue
                if cmd_type == "start":
                    camera.start_acquisition()
                    should_stream = True
                    last_frame_time = time.time()
                    publish_status(frame_queue, camera.get_status())
                    continue
                if cmd_type == "pause":
                    camera.stop_acquisition()
                    should_stream = False
                    publish_status(frame_queue, camera.get_status())
                    continue
                if cmd_type == "connect":
                    connect_camera(command.get("mode", mode))
                    should_stream = camera is not None and camera.status.acquiring
                    last_frame_time = time.time()
                    continue
                if cmd_type == "disconnect":
                    # Manual disconnect: stop the stream + release the device and
                    # STAY offline (should_stream=False disables auto-reconnect) so
                    # a split/desynced stream can be cleanly re-established on the
                    # user's next Connect / Reconnect.
                    if camera is not None:
                        camera.stop_acquisition()
                        camera.disconnect()
                        publish_status(frame_queue, camera.get_status())
                    should_stream = False
                    continue
                if cmd_type == "read_temp":
                    if hasattr(camera, "refresh_temperatures"):
                        try:
                            camera.refresh_temperatures()
                        except Exception:  # noqa: BLE001
                            pass
                    publish_status(frame_queue, camera.get_status())
                    continue
                if cmd_type == "snapshot":
                    frame = camera.get_frame()
                    if frame is not None:
                        publish_frame(frame_queue, frame, shared_frame)
                    continue
            except queue.Empty:
                pass

            loop_start = time.time()
            frame = camera.get_frame()
            if frame is not None:
                last_frame_time = loop_start
                publish_frame(frame_queue, frame, shared_frame)
            else:
                now = time.time()
                if now - last_status_push > 1.0:
                    publish_status(frame_queue, camera.get_status())
                    last_status_push = now

                # Auto-reconnect: if we intend to stream but no frame has arrived
                # for a while, the GigE link dropped (NOT_CONNECTED). Keep trying
                # (throttled) until it returns -- self-healing across link drops.
                if (should_stream
                        and now - last_frame_time > RECONNECT_AFTER_S
                        and now - last_reconnect_try > RECONNECT_THROTTLE_S):
                    last_reconnect_try = now
                    if hasattr(camera, "reconnect"):
                        logger.warning("No frames -- attempting camera reconnect")
                        try:
                            camera.reconnect()
                        except Exception:  # noqa: BLE001
                            logger.exception("reconnect attempt failed")
                        publish_status(frame_queue, camera.get_status())
                        last_frame_time = time.time()   # grace before next try

            # Pace the loop. get_frame() already blocks at the camera's native
            # frame period (RetrieveBuffer), so only sleep for the *remaining*
            # slice of the target interval -- never add latency on top of the
            # hardware rate. With a high target_fps the camera sets the rate.
            if frame is None:
                time.sleep(0.005)
            else:
                remaining = frame_interval - (time.time() - loop_start)
                if remaining > 0:
                    time.sleep(remaining)
    except Exception:
        logger.exception("Camera worker crashed")
    finally:
        if camera is not None:
            camera.stop_acquisition()
            camera.disconnect()
            publish_status(frame_queue, camera.get_status())
        if shared_memory_handle is not None:
            shared_memory_handle.close()
        logger.info("Camera worker exited")

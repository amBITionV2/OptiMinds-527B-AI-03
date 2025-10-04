// VideoFrameExtractorFS.jsx
import React, { useRef, useState, useEffect } from "react";

/**
 * VideoFrameExtractorFS
 *
 * - Uses a number input to set FPS and a dropdown for image format.
 * - Uses File System Access API to save frames locally.
 *
 * Browser compatibility: works best on Chromium-based browsers (Chrome/Edge/Opera).
 */
export default function VideoFrameExtractorFS() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  const [videoSrc, setVideoSrc] = useState(null);
  const [videoFileName, setVideoFileName] = useState("");
  const [label, setLabel] = useState("low");
  const [dirHandle, setDirHandle] = useState(null);
  const [processing, setProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [message, setMessage] = useState("Select a video file to begin.");
  const [durationSeconds, setDurationSeconds] = useState(0);

  // Replaced slider with numeric input
  const [frameRate, setFrameRate] = useState(10);
  const [imageFormat, setImageFormat] = useState("png"); // 'png', 'jpeg', or 'webp'

  const fsApiAvailable = typeof window.showDirectoryPicker === "function";

  useEffect(() => {
    return () => {
      if (videoSrc) {
        URL.revokeObjectURL(videoSrc);
      }
    };
  }, [videoSrc]);

  function seekVideo(videoEl, time) {
    return new Promise((resolve, reject) => {
      const onSeeked = () => setTimeout(resolve, 50);
      const onError = (e) => reject(new Error("Video seek error: " + e?.message));
      videoEl.addEventListener("seeked", onSeeked, { once: true });
      videoEl.addEventListener("error", onError, { once: true });
      videoEl.currentTime = time;
    });
  }

  function captureFrameBlob(videoEl, mimeType = "image/png", quality = 0.9) {
    const canvas = canvasRef.current;
    if (!canvas) return Promise.reject(new Error("Canvas not found"));
    canvas.width = videoEl.videoWidth;
    canvas.height = videoEl.videoHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(videoEl, 0, 0, canvas.width, canvas.height);
    return new Promise((resolve) => {
      canvas.toBlob((blob) => resolve(blob), mimeType, quality);
    });
  }

  async function handleChooseDirectory() {
    try {
      const handle = await window.showDirectoryPicker();
      setDirHandle(handle);
      setMessage(`Selected directory: ${handle.name}`);
    } catch (err) {
      console.warn("Directory pick cancelled or failed", err);
      setMessage("Directory selection was cancelled.");
    }
  }

  async function writeBlobToFile(dirHandle, filename, blob) {
    const fileHandle = await dirHandle.getFileHandle(filename, { create: true });
    const writable = await fileHandle.createWritable();
    await writable.write(blob);
    await writable.close();
  }

  async function processVideoToFs() {
    if (!videoRef.current?.duration) {
      setMessage("Video is not loaded properly.");
      return;
    }
    if (!dirHandle) {
      setMessage("Please choose a directory to save the frames.");
      return;
    }

    setProcessing(true);
    setProgress(0);
    setMessage("Processing...");

    const videoEl = videoRef.current;
    const totalDuration = videoEl.duration;
    const timeIncrement = 1 / frameRate;
    const mimeType = `image/${imageFormat}`;
    const extension = imageFormat === "jpeg" ? "jpg" : imageFormat;
    let frameCounter = 0;

    if (totalDuration <= 0) {
      setMessage("Video has no duration. Cannot process.");
      setProcessing(false);
      return;
    }

    let labelDir;
    try {
      labelDir = await dirHandle.getDirectoryHandle(label, { create: true });
    } catch (err) {
      setMessage(`Could not create folder for label: ${err.message}`);
      setProcessing(false);
      return;
    }

    for (let t = 0; t <= totalDuration; t += timeIncrement) {
      try {
        await seekVideo(videoEl, t);
        const blob = await captureFrameBlob(videoEl, mimeType);
        if (blob) {
          const filename = `${label}_${String(frameCounter).padStart(5, "0")}.${extension}`;
          await writeBlobToFile(labelDir, filename, blob);
        }
        setProgress(Math.round((t / totalDuration) * 100));
        frameCounter++;
      } catch (err) {
        console.error(`Error saving frame at ${t.toFixed(2)}s:`, err);
      }
    }

    setProcessing(false);
    setMessage(`Done! Saved ${frameCounter} frames to folder: "${label}"`);
    setProgress(100);
  }

  function handleVideoFileChange(e) {
    const file = e.target.files?.[0];
    if (!file) {
      setVideoSrc(null);
      setVideoFileName("");
      return;
    }
    if (videoSrc) URL.revokeObjectURL(videoSrc);
    const objectUrl = URL.createObjectURL(file);
    setVideoSrc(objectUrl);
    setVideoFileName(file.name);
    setProgress(0);
    setMessage(`Loading video: ${file.name}`);
  }

  function handleMetadataLoaded() {
    const video = videoRef.current;
    if (!video) return;
    const duration = Math.floor(video.duration);
    setDurationSeconds(duration);
    setMessage(`Video ready. Duration: ${duration}s. Configure and extract frames.`);
  }

  return (
    <div style={{ maxWidth: 900, margin: "16px auto", fontFamily: "system-ui, Arial" }}>
      <h2>Video Frame Extractor</h2>
      <p>This tool extracts frames from a video and saves them to a local folder using the File System Access API.</p>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "auto 1fr",
          gap: "16px 12px",
          alignItems: "center",
          background: "#f9f9f9",
          padding: "16px",
          borderRadius: "8px",
        }}
      >
        {/* Row 1 */}
        <strong style={{ textAlign: "right" }}>1. Video File:</strong>
        <input type="file" accept="video/*" onChange={handleVideoFileChange} />

        {/* Row 2 */}
        <strong style={{ textAlign: "right" }}>2. Frame Rate (FPS):</strong>
        <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
          <input
            type="number"
            min="1"
            max="60"
            value={frameRate}
            onChange={(e) => setFrameRate(Number(e.target.value))}
            style={{ width: "80px", padding: "4px" }}
          />
          <span>frames/sec</span>
        </div>

        {/* Row 3 */}
        <strong style={{ textAlign: "right" }}>3. Image Format:</strong>
        <select value={imageFormat} onChange={(e) => setImageFormat(e.target.value)}>
          <option value="png">PNG</option>
          <option value="jpeg">JPEG (.jpg)</option>
          <option value="webp">WebP</option>
        </select>

        {/* Row 4 */}
        <strong style={{ textAlign: "right" }}>4. Label:</strong>
        <select value={label} onChange={(e) => setLabel(e.target.value)}>
          <option value="low">low</option>
          <option value="medium">medium</option>
          <option value="high">high</option>
          <option value="unlabeled">unlabeled</option>
        </select>

        {/* Row 5 */}
        <strong style={{ textAlign: "right" }}>5. Save Directory:</strong>
        <button onClick={handleChooseDirectory} disabled={!fsApiAvailable}>
          {dirHandle ? `Directory: ${dirHandle.name}` : "Choose Directory"}
        </button>

        {/* Row 6 */}
        <strong style={{ textAlign: "right" }}>6. Extract:</strong>
        <button
          onClick={processVideoToFs}
          disabled={processing || !videoSrc || !dirHandle}
          style={{
            background: "#0b7",
            color: "white",
            border: "none",
            padding: "8px 12px",
            borderRadius: "4px",
            cursor: "pointer",
          }}
        >
          {processing ? `Processing... ${progress}%` : `Extract & Save Frames`}
        </button>
      </div>

      <div style={{ marginTop: 16 }}>
        <div style={{ fontWeight: "bold" }}>Status:</div>
        <div style={{ marginTop: 4, color: "#333", minHeight: "20px" }}>{message}</div>
        {(processing || progress === 100) && (
          <progress value={progress} max="100" style={{ width: "100%", marginTop: "8px" }}></progress>
        )}
      </div>

      <div style={{ marginTop: 18 }}>
        <video
          ref={videoRef}
          src={videoSrc}
          onLoadedMetadata={handleMetadataLoaded}
          controls
          style={{ width: "100%", background: "#eee", display: videoSrc ? "block" : "none" }}
        />
        <canvas ref={canvasRef} style={{ display: "none" }} />
      </div>

      {!fsApiAvailable && (
        <div
          style={{
            marginTop: 12,
            padding: "12px",
            background: "#fff0f0",
            color: "crimson",
            border: "1px solid crimson",
            borderRadius: "4px",
          }}
        >
          <strong>Browser Not Supported:</strong> Your browser does not support the File System Access API. Please use a
          modern Chromium-based browser like Chrome or Edge.
        </div>
      )}
    </div>
  );
}

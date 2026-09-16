import glob
import os
import subprocess
import time
from playwright.sync_api import sync_playwright

SCREENSHOTS_DIR = os.path.abspath("docs/screenshots")
RAW_VIDEO_DIR = os.path.join(SCREENSHOTS_DIR, "raw_video")
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
os.makedirs(RAW_VIDEO_DIR, exist_ok=True)

# Clean out old raw webms and screenshot 08 if present
for old_webm in glob.glob(os.path.join(RAW_VIDEO_DIR, "*.webm")):
    try:
        os.remove(old_webm)
    except Exception:
        pass

old_08 = os.path.join(SCREENSHOTS_DIR, "08_hemisphere_split.png")
if os.path.exists(old_08):
    try:
        os.remove(old_08)
    except Exception:
        pass

print("Starting complete walkthrough recording (1-7, then 9-11)...", flush=True)

with sync_playwright() as p:
    browser = p.chromium.launch(
        channel="msedge",
        headless=True,
        args=[
            "--use-gl=angle",
            "--use-angle=d3d11",
            "--enable-webgl",
            "--no-sandbox",
            "--disable-setuid-sandbox",
        ]
    )

    context = browser.new_context(
        viewport={"width": 1600, "height": 900},
        record_video_dir=RAW_VIDEO_DIR,
        record_video_size={"width": 1600, "height": 900},
    )

    page = context.new_page()

    # Step 1: Landing welcome page
    print("Step 1: Navigating to http://localhost:5050...", flush=True)
    page.goto("http://localhost:5050", wait_until="networkidle")
    time.sleep(2.0)
    page.screenshot(path=os.path.join(SCREENSHOTS_DIR, "01_landing_welcome.png"))
    print("Saved 01_landing_welcome.png", flush=True)

    # Step 2: Load Demo Scan and wait for WebGL to fully render
    print("Step 2: Clicking Demo Scan button...", flush=True)
    page.click("#btn-demo")
    page.wait_for_selector("#overlay.hidden", timeout=30000)
    page.wait_for_function("""() => {
        const status = document.getElementById('h-status')?.textContent;
        return status === 'ACTIVE' && window.nv1 && window.nv1.volumes && window.nv1.volumes.length > 0;
    }""", timeout=30000)
    time.sleep(4.0)  # Allow WebGL shaders to complete initial render pass
    page.screenshot(path=os.path.join(SCREENSHOTS_DIR, "02_demo_loaded_ortho.png"))
    print("Saved 02_demo_loaded_ortho.png", flush=True)

    # Step 3: Interactive Display Controls (Thermal Colormap)
    print("Step 3: Selecting Thermal Colormap...", flush=True)
    page.click(".cb:has-text('THERMAL')")
    time.sleep(3.0)
    page.screenshot(path=os.path.join(SCREENSHOTS_DIR, "03_display_colormap_hot.png"))
    print("Saved 03_display_colormap_hot.png", flush=True)
    page.click(".cb:has-text('GRAYSCALE')")
    time.sleep(1.5)

    # Step 4: FOV Localizer Modal - wait until all 3 mini viewers are loaded & rendered
    print("Step 4: Opening FOV Localizer and waiting for all 3 views...", flush=True)
    page.wait_for_function("""() => {
        return typeof _fovOrigPath !== 'undefined' && _fovOrigPath !== null;
    }""", timeout=15000)

    page.click("#btn-fov-adjust")
    page.wait_for_selector("#fov-modal", state="visible", timeout=10000)

    print("Waiting for mini viewers to render brain slices...", flush=True)
    page.wait_for_function("""() => {
        const ready = window._fovNVReady;
        const nvs = window._fovNV;
        const status = document.getElementById('fov-adj-status')?.textContent || '';
        return ready &&
               ready.length === 3 &&
               ready.every(Boolean) &&
               nvs &&
               nvs[0]?.volumes?.length > 0 &&
               nvs[1]?.volumes?.length > 0 &&
               nvs[2]?.volumes?.length > 0 &&
               !status.includes('Loading');
    }""", timeout=30000)

    page.evaluate("""() => {
        if (window._fovNV) window._fovNV.forEach(nv => { try { nv.drawScene(); } catch(e) {} });
    }""")
    time.sleep(3.5)
    page.screenshot(path=os.path.join(SCREENSHOTS_DIR, "04_fov_crop_inspection.png"))
    print("Saved 04_fov_crop_inspection.png", flush=True)
    page.click("#fov-btn-cancel")
    page.wait_for_selector("#fov-modal", state="hidden", timeout=5000)
    time.sleep(1.5)

    # Step 5: Segmentation Options
    print("Step 5: Configuring Segmentation Options...", flush=True)
    page.click("#seg-opts-toggle")
    time.sleep(0.5)
    page.check("#opt-fast")
    page.check("#opt-low-memory")
    time.sleep(1.5)
    page.screenshot(path=os.path.join(SCREENSHOTS_DIR, "05_segmentation_options.png"))
    print("Saved 05_segmentation_options.png", flush=True)

    # Step 6: Click Run Segmentation - Reseg dialog
    print("Step 6: Triggering Run Segmentation dialog...", flush=True)
    page.click("#btn-seg")
    page.wait_for_selector("#reseg-modal.on", timeout=15000)
    time.sleep(1.5)
    page.screenshot(path=os.path.join(SCREENSHOTS_DIR, "06_run_segmentation_dialog.png"))
    print("Saved 06_run_segmentation_dialog.png", flush=True)

    # Step 7: Load Existing Segmentation and wait for ALL tissue overlays to render
    print("Step 7: Loading Existing Segmentation & waiting for all overlays...", flush=True)
    page.click("#reseg-modal button:has-text('LOAD EXISTING')")
    page.wait_for_selector("#export3d-wrap.on", timeout=30000)

    print("Overlays confirmed loaded by server. Drawing WebGL scenes...", flush=True)
    page.evaluate("""() => {
        [window.nv1, window.nv2, window.nv3, window.nv4].forEach(nv => {
            try { nv.drawScene(); } catch(e) {}
        });
    }""")
    time.sleep(5.0)
    page.screenshot(path=os.path.join(SCREENSHOTS_DIR, "07_segmentation_loaded_tissue.png"))
    print("Saved 07_segmentation_loaded_tissue.png", flush=True)

    # Step 9: 3D Surface / Mesh Preview Modal (Skipping 8 as requested)
    print("Step 9: Opening 3D Export Dialog & waiting for 3D render to finish...", flush=True)
    page.evaluate("""() => {
        document.getElementById('fmt-stl').checked = true;
        const btn = document.getElementById('btn-export3d');
        btn.disabled = false;
        onExport3dClick();
    }""")

    page.wait_for_function("""() => {
        const el = document.getElementById('p3d-loading');
        return el && (el.style.display === 'none' || el.offsetParent === null);
    }""", timeout=120000)

    page.evaluate("""() => {
        if (window._p3dNV) {
            window._p3dNV.scene.renderAzimuth = 285;
            window._p3dNV.scene.renderElevation = 25;
            window._p3dNV.drawScene();
        }
    }""")
    time.sleep(4.5)
    page.screenshot(path=os.path.join(SCREENSHOTS_DIR, "09_export_3d_preview.png"))
    print("Saved 09_export_3d_preview.png", flush=True)

    # Step 10: STL Export Settings Modal
    print("Step 10: Opening STL Export Settings...", flush=True)
    page.click("#p3d-btn-accept")
    page.wait_for_selector("#stl-modal.on", timeout=10000)
    page.evaluate("""() => {
        const sl = document.getElementById('stl-faces');
        if (sl) sl.value = '100000';
    }""")
    time.sleep(2.0)
    page.screenshot(path=os.path.join(SCREENSHOTS_DIR, "10_stl_export_settings.png"))
    print("Saved 10_stl_export_settings.png", flush=True)

    # Step 11: Execute STL Export and wait for STL 3D viewer modal
    print("Step 11: Confirming STL Export and waiting for STL viewer...", flush=True)
    page.click(".stl-btns button:has-text('EXPORT')")
    print("Waiting for STL export to complete and 3D viewer modal to open...", flush=True)
    page.wait_for_selector("#stl-viewer-modal.on", timeout=120000)

    # Wait until Three.js finishes fetching mesh and renders
    page.wait_for_function("""() => {
        const info = document.getElementById('stl-viewer-info')?.textContent || '';
        return info.includes('triangles');
    }""", timeout=35000)

    time.sleep(4.5)
    page.screenshot(path=os.path.join(SCREENSHOTS_DIR, "11_stl_exported_viewer.png"))
    print("Saved 11_stl_exported_viewer.png", flush=True)

    time.sleep(3.0)
    context.close()
    browser.close()

print("Browser session closed. Encoding walkthrough video...", flush=True)

# Convert latest webm to mp4 and animated gif
webms = glob.glob(os.path.join(RAW_VIDEO_DIR, "*.webm"))
if webms:
    latest_webm = sorted(webms, key=os.path.getmtime, reverse=True)[0]
    import imageio_ffmpeg
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    mp4_out = os.path.join(SCREENSHOTS_DIR, "neuroflux_demo.mp4")
    gif_out = os.path.join(SCREENSHOTS_DIR, "neuroflux_demo.gif")

    print(f"Converting {latest_webm} to MP4...", flush=True)
    subprocess.run([
        ffmpeg_exe, "-y", "-i", latest_webm,
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "22", "-preset", "fast",
        mp4_out
    ], check=True)
    print(f"Generated {mp4_out}", flush=True)

    print(f"Converting {latest_webm} to animated GIF...", flush=True)
    subprocess.run([
        ffmpeg_exe, "-y", "-i", latest_webm,
        "-vf", "fps=8,scale=960:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer",
        gif_out
    ], check=True)
    print(f"Generated {gif_out}", flush=True)

print("WALKTHROUGH RECORDING & SCREENSHOTS FULLY GENERATED!", flush=True)

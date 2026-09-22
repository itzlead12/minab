/**
 * Minab 3D Reconstruction Viewer
 * Renders Monocular 3D Meshes, Point Clouds, Camera Trajectories, and Frustums using Three.js
 * Designed according to instruction.md specifications for orbit-based monocular reconstruction.
 */

class Minab3DViewer {
    constructor(containerId, options = {}) {
        this.container = document.getElementById(containerId);
        if (!this.container) return;

        this.resultId = options.resultId;
        this.meshPlyUrl = options.meshPlyUrl;
        this.meshObjUrl = options.meshObjUrl || options.meshUrl;
        this.pointCloudUrl = options.pointCloudUrl;
        this.trajectoryJsonUrl = options.trajectoryJsonUrl;
        this.trajectoryTxtUrl = options.trajectoryUrl;

        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;

        this.meshObject = null;
        this.pointCloudObject = null;
        this.trajectoryObject = null;
        this.frustumsGroup = null;
        this.gridHelper = null;
        this.axesHelper = null;

        this.showMesh = true;
        this.showPointCloud = true;
        this.showTrajectory = true;
        this.showFrustums = true;
        this.showGrid = true;
        this.wireframeMode = false;
        this.pointSize = 0.025;

        this.initScene();
        this.loadArtifacts();
        this.animate();

        window.addEventListener('resize', () => this.onWindowResize());
    }

    initScene() {
        const width = this.container.clientWidth || 800;
        const height = this.container.clientHeight || 520;

        // Scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x080c18); // deep slate studio bg

        // Camera
        this.camera = new THREE.PerspectiveCamera(55, width / height, 0.01, 1000);
        this.camera.position.set(0.0, 1.2, 3.2);

        // WebGL Renderer
        this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: "high-performance" });
        this.renderer.setSize(width, height);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.renderer.outputEncoding = THREE.sRGBEncoding;

        // Clear container
        this.container.innerHTML = "";
        this.container.appendChild(this.renderer.domElement);

        // Orbit Controls
        if (THREE.OrbitControls) {
            this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
            this.controls.enableDamping = true;
            this.controls.dampingFactor = 0.06;
            this.controls.screenSpacePanning = true;
            this.controls.maxDistance = 25;
            this.controls.minDistance = 0.1;
        }

        // Studio Lighting
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.75);
        this.scene.add(ambientLight);

        const keyLight = new THREE.DirectionalLight(0xffffff, 0.85);
        keyLight.position.set(5, 8, 5);
        this.scene.add(keyLight);

        const fillLight = new THREE.DirectionalLight(0x60a5fa, 0.45);
        fillLight.position.set(-5, -3, -5);
        this.scene.add(fillLight);

        const rimLight = new THREE.DirectionalLight(0x38bdf8, 0.3);
        rimLight.position.set(0, -6, 4);
        this.scene.add(rimLight);

        // Studio Ground Grid
        this.gridHelper = new THREE.GridHelper(8, 24, 0x3b82f6, 0x1e293b);
        this.gridHelper.position.y = -0.5;
        this.scene.add(this.gridHelper);

        this.axesHelper = new THREE.AxesHelper(0.5);
        this.axesHelper.position.set(0, -0.49, 0);
        this.scene.add(this.axesHelper);
    }

    async loadArtifacts() {
        // 1. Try loading colored PLY mesh first, fallback to OBJ
        if (this.meshPlyUrl) {
            this.loadMeshPly(this.meshPlyUrl);
        } else if (this.meshObjUrl) {
            this.loadMeshObj(this.meshObjUrl);
        }

        // 2. Load dense point cloud
        if (this.pointCloudUrl) {
            this.loadPointCloud(this.pointCloudUrl);
        }

        // 3. Load trajectory and camera frustums (JSON preferred, fallback to TXT)
        if (this.trajectoryJsonUrl) {
            this.loadTrajectoryJson(this.trajectoryJsonUrl);
        } else if (this.trajectoryTxtUrl) {
            this.loadTrajectoryTxt(this.trajectoryTxtUrl);
        }
    }

    loadMeshPly(url) {
        if (typeof THREE.PLYLoader === 'undefined') {
            if (this.meshObjUrl) this.loadMeshObj(this.meshObjUrl);
            return;
        }

        const loader = new THREE.PLYLoader();
        loader.load(
            url,
            (geometry) => {
                geometry.computeVertexNormals();
                const hasColors = geometry.attributes.color !== undefined;

                const material = new THREE.MeshStandardMaterial({
                    vertexColors: hasColors,
                    color: hasColors ? 0xffffff : 0xd1d5db,
                    roughness: 0.45,
                    metalness: 0.08,
                    side: THREE.DoubleSide,
                    wireframe: this.wireframeMode
                });

                this.meshObject = new THREE.Mesh(geometry, material);
                this.scene.add(this.meshObject);
                this.updateMeshStats(geometry);
            },
            undefined,
            (err) => {
                console.warn("PLY mesh load failed, falling back to OBJ:", err);
                if (this.meshObjUrl) this.loadMeshObj(this.meshObjUrl);
            }
        );
    }

    loadMeshObj(url) {
        if (typeof THREE.OBJLoader === 'undefined') return;

        const loader = new THREE.OBJLoader();
        loader.load(
            url,
            (obj) => {
                const material = new THREE.MeshStandardMaterial({
                    color: 0xe2e8f0,
                    roughness: 0.45,
                    metalness: 0.1,
                    side: THREE.DoubleSide,
                    wireframe: this.wireframeMode
                });

                obj.traverse((child) => {
                    if (child.isMesh) {
                        child.material = material;
                        child.geometry.computeVertexNormals();
                        this.updateMeshStats(child.geometry);
                    }
                });

                this.meshObject = obj;
                this.scene.add(this.meshObject);
            },
            undefined,
            (err) => console.warn("OBJ mesh load failed:", err)
        );
    }

    loadPointCloud(url) {
        if (typeof THREE.PLYLoader === 'undefined') return;

        const loader = new THREE.PLYLoader();
        loader.load(
            url,
            (geometry) => {
                geometry.computeVertexNormals();
                const hasColors = geometry.attributes.color !== undefined;

                const material = new THREE.PointsMaterial({
                    size: this.pointSize,
                    vertexColors: hasColors,
                    color: hasColors ? 0xffffff : 0x38bdf8,
                    sizeAttenuation: true
                });

                this.pointCloudObject = new THREE.Points(geometry, material);
                this.scene.add(this.pointCloudObject);

                // Auto-align ground grid to bottom of scene
                geometry.computeBoundingBox();
                if (geometry.boundingBox) {
                    const minY = geometry.boundingBox.min.y;
                    this.gridHelper.position.y = minY - 0.02;
                    this.axesHelper.position.y = minY - 0.02;
                }

                this.updatePointStats(geometry);
            },
            undefined,
            (err) => console.warn("PointCloud PLY load failed:", err)
        );
    }

    createCameraFrustumGeometry(scale = 0.12) {
        const w = scale * 0.7;
        const h = scale * 0.5;
        const z = scale;

        // Camera apex at origin (0,0,0), base rectangle at +z
        const vertices = new Float32Array([
            // Rays from apex to base corners
            0, 0, 0,  -w, -h, z,
            0, 0, 0,   w, -h, z,
            0, 0, 0,   w,  h, z,
            0, 0, 0,  -w,  h, z,
            // Perimeter rectangle
            -w, -h, z,   w, -h, z,
             w, -h, z,   w,  h, z,
             w,  h, z,  -w,  h, z,
            -w,  h, z,  -w, -h, z,
            // Top orientation tick (indicates camera UP direction)
            0, -h, z,   0, -h - (h * 0.4), z
        ]);

        const geom = new THREE.BufferGeometry();
        geom.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
        return geom;
    }

    async loadTrajectoryJson(url) {
        try {
            const resp = await fetch(url);
            if (!resp.ok) {
                if (this.trajectoryTxtUrl) this.loadTrajectoryTxt(this.trajectoryTxtUrl);
                return;
            }
            const data = await resp.json();
            const poses = data.poses || [];
            if (!poses || poses.length === 0) return;

            const points = [];
            this.frustumsGroup = new THREE.Group();
            const frustumGeom = this.createCameraFrustumGeometry(0.14);

            poses.forEach((pose, idx) => {
                const pos = new THREE.Vector3(pose.position[0], pose.position[1], pose.position[2]);
                points.push(pos);

                // Sample frustums periodically along arc
                if (idx % 2 === 0 || idx === poses.length - 1) {
                    let color = 0x06b6d4; // Cyan intermediate
                    if (idx === 0) color = 0x22c55e; // Green start
                    else if (idx === poses.length - 1) color = 0xeab308; // Yellow end

                    const frustumMat = new THREE.LineBasicMaterial({ color: color, linewidth: 2 });
                    const frustum = new THREE.LineSegments(frustumGeom, frustumMat);
                    frustum.position.copy(pos);

                    if (pose.quaternion && pose.quaternion.length === 4) {
                        const q = pose.quaternion;
                        frustum.quaternion.set(q[0], q[1], q[2], q[3]);
                    }
                    this.frustumsGroup.add(frustum);
                }
            });

            // Trajectory polyline
            const curveGeom = new THREE.BufferGeometry().setFromPoints(points);
            const lineMat = new THREE.LineBasicMaterial({ color: 0xec4899, linewidth: 2.5 });
            const line = new THREE.Line(curveGeom, lineMat);

            this.trajectoryObject = new THREE.Group();
            this.trajectoryObject.add(line);
            this.scene.add(this.trajectoryObject);
            this.scene.add(this.frustumsGroup);

            this.updateTrajectoryStats(poses.length);
        } catch (e) {
            console.warn("Failed to load trajectory JSON, falling back to TXT:", e);
            if (this.trajectoryTxtUrl) this.loadTrajectoryTxt(this.trajectoryTxtUrl);
        }
    }

    async loadTrajectoryTxt(url) {
        try {
            const resp = await fetch(url);
            if (!resp.ok) return;
            const text = await resp.text();
            const lines = text.split('\n');

            const points = [];
            lines.forEach(line => {
                line = line.trim();
                if (!line || line.startsWith('#')) return;
                const parts = line.split(/\s+/);
                if (parts.length >= 4) {
                    const tx = parseFloat(parts[1]);
                    const ty = parseFloat(parts[2]);
                    const tz = parseFloat(parts[3]);
                    if (!isNaN(tx) && !isNaN(ty) && !isNaN(tz)) {
                        points.push(new THREE.Vector3(tx, ty, tz));
                    }
                }
            });

            if (points.length > 0) {
                const geometry = new THREE.BufferGeometry().setFromPoints(points);
                const material = new THREE.LineBasicMaterial({ color: 0xec4899, linewidth: 2 });
                const line = new THREE.Line(geometry, material);

                this.trajectoryObject = new THREE.Group();
                this.trajectoryObject.add(line);
                this.scene.add(this.trajectoryObject);
                this.updateTrajectoryStats(points.length);
            }
        } catch (e) {
            console.warn("Trajectory loading error:", e);
        }
    }

    toggleMesh(visible) {
        this.showMesh = visible;
        if (this.meshObject) this.meshObject.visible = visible;
    }

    togglePointCloud(visible) {
        this.showPointCloud = visible;
        if (this.pointCloudObject) this.pointCloudObject.visible = visible;
    }

    toggleTrajectory(visible) {
        this.showTrajectory = visible;
        if (this.trajectoryObject) this.trajectoryObject.visible = visible;
    }

    toggleFrustums(visible) {
        this.showFrustums = visible;
        if (this.frustumsGroup) this.frustumsGroup.visible = visible;
    }

    toggleGrid(visible) {
        this.showGrid = visible;
        if (this.gridHelper) this.gridHelper.visible = visible;
        if (this.axesHelper) this.axesHelper.visible = visible;
    }

    setWireframe(enabled) {
        this.wireframeMode = enabled;
        if (this.meshObject) {
            if (this.meshObject.material) {
                this.meshObject.material.wireframe = enabled;
            } else {
                this.meshObject.traverse((c) => {
                    if (c.isMesh && c.material) c.material.wireframe = enabled;
                });
            }
        }
    }

    setPointSize(size) {
        this.pointSize = size;
        if (this.pointCloudObject && this.pointCloudObject.material) {
            this.pointCloudObject.material.size = size;
        }
    }

    resetView() {
        if (!this.camera || !this.controls) return;
        this.camera.position.set(0.0, 1.2, 3.2);
        this.controls.target.set(0, 0, 0);
        this.controls.update();
    }

    topView() {
        if (!this.camera || !this.controls) return;
        this.camera.position.set(0.0, 3.5, 0.01);
        this.controls.target.set(0, 0, 0);
        this.controls.update();
    }

    frontView() {
        if (!this.camera || !this.controls) return;
        this.camera.position.set(0.0, 0.0, 3.2);
        this.controls.target.set(0, 0, 0);
        this.controls.update();
    }

    updatePointStats(geometry) {
        if (geometry && geometry.attributes.position) {
            const count = geometry.attributes.position.count;
            const el = document.getElementById('stat-points-count');
            if (el) el.innerText = count.toLocaleString();
        }
    }

    updateMeshStats(geometry) {
        if (geometry && geometry.index) {
            const triCount = geometry.index.count / 3;
            const el = document.getElementById('stat-triangles-count');
            if (el) el.innerText = Math.round(triCount).toLocaleString();
        }
    }

    updateTrajectoryStats(count) {
        const el = document.getElementById('stat-trajectory-count');
        if (el) el.innerText = `${count} poses`;
    }

    onWindowResize() {
        if (!this.container || !this.renderer || !this.camera) return;
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);
    }

    animate() {
        requestAnimationFrame(() => this.animate());
        if (this.controls) this.controls.update();
        if (this.renderer && this.scene && this.camera) {
            this.renderer.render(this.scene, this.camera);
        }
    }
}

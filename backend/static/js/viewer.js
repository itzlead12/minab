/**
 * Minab 3D Reconstruction Studio Viewer
 * Three.js Viewer supporting Meshes (PLY/OBJ), Dense Point Clouds (PLY),
 * Camera Trajectories, and Frustums in Arbitrary Relative Coordinates.
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
        this.showAxes = true;
        this.wireframeMode = false;
        this.pointSize = 0.025;

        this.initScene();
        this.loadArtifacts();
        this.animate();

        window.addEventListener('resize', () => this.onWindowResize());
    }

    initScene() {
        const width = this.container.clientWidth || 800;
        const height = this.container.clientHeight || 540;

        // Scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x05070a);

        // Camera
        this.camera = new THREE.PerspectiveCamera(50, width / height, 0.01, 1000);
        this.camera.position.set(0.0, 1.2, 2.4);

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
            this.controls.target.set(0, 0, 0);
            this.controls.maxDistance = 25;
            this.controls.minDistance = 0.05;
        }

        // Studio Lighting
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.85);
        this.scene.add(ambientLight);

        const keyLight = new THREE.DirectionalLight(0xffffff, 0.9);
        keyLight.position.set(4, 6, 4);
        this.scene.add(keyLight);

        const fillLight = new THREE.DirectionalLight(0x3b82f6, 0.4);
        fillLight.position.set(-4, -2, -4);
        this.scene.add(fillLight);

        const rimLight = new THREE.DirectionalLight(0x60a5fa, 0.35);
        rimLight.position.set(0, 5, -5);
        this.scene.add(rimLight);

        // Studio Ground Grid
        this.gridHelper = new THREE.GridHelper(6, 24, 0x2563eb, 0x18232f);
        this.gridHelper.position.y = -0.5;
        this.scene.add(this.gridHelper);

        this.axesHelper = new THREE.AxesHelper(0.4);
        this.axesHelper.position.set(0, -0.49, 0);
        this.scene.add(this.axesHelper);
    }

    loadArtifacts() {
        if (this.meshPlyUrl) {
            this.loadMeshPly(this.meshPlyUrl);
        } else if (this.meshObjUrl) {
            this.loadMeshObj(this.meshObjUrl);
        }

        if (this.pointCloudUrl) {
            this.loadPointCloud(this.pointCloudUrl);
        }

        if (this.trajectoryJsonUrl) {
            this.loadTrajectoryJson(this.trajectoryJsonUrl);
        } else if (this.trajectoryTxtUrl) {
            this.loadTrajectoryTxt(this.trajectoryTxtUrl);
        }
    }

    loadMeshPly(url) {
        if (!THREE.PLYLoader) return;
        const loader = new THREE.PLYLoader();
        loader.load(url, (geometry) => {
            geometry.computeVertexNormals();

            if (this.meshObject) {
                this.scene.remove(this.meshObject);
            }

            let material;
            if (geometry.hasAttribute('color')) {
                material = new THREE.MeshStandardMaterial({
                    vertexColors: true,
                    roughness: 0.5,
                    metalness: 0.1,
                    wireframe: this.wireframeMode,
                    side: THREE.DoubleSide
                });
            } else {
                material = new THREE.MeshStandardMaterial({
                    color: 0x94a3b8,
                    roughness: 0.6,
                    metalness: 0.2,
                    wireframe: this.wireframeMode,
                    side: THREE.DoubleSide
                });
            }
            this.meshObject = new THREE.Mesh(geometry, material);
            this.meshObject.visible = this.showMesh;
            this.scene.add(this.meshObject);
            this.updateMeshStats(geometry);
        }, undefined, (e) => {
            console.warn("PLY mesh load failed, trying OBJ:", e);
            if (this.meshObjUrl) this.loadMeshObj(this.meshObjUrl);
        });
    }

    loadMeshObj(url) {
        if (!THREE.OBJLoader) return;
        const loader = new THREE.OBJLoader();
        loader.load(url, (object) => {
            if (this.meshObject) {
                this.scene.remove(this.meshObject);
            }
            this.meshObject = object;
            object.traverse((child) => {
                if (child.isMesh) {
                    child.material = new THREE.MeshStandardMaterial({
                        color: 0x94a3b8,
                        roughness: 0.5,
                        metalness: 0.1,
                        wireframe: this.wireframeMode,
                        side: THREE.DoubleSide
                    });
                }
            });
            this.meshObject.visible = this.showMesh;
            this.scene.add(this.meshObject);
        }, undefined, (err) => {
            console.debug("OBJ mesh load notice:", err);
        });
    }

    loadPointCloud(url) {
        if (!THREE.PLYLoader) return;
        const loader = new THREE.PLYLoader();
        loader.load(url, (geometry) => {
            if (this.pointCloudObject) {
                this.scene.remove(this.pointCloudObject);
            }

            let material;
            if (geometry.hasAttribute('color')) {
                material = new THREE.PointsMaterial({
                    size: this.pointSize,
                    vertexColors: true,
                    sizeAttenuation: true
                });
            } else {
                material = new THREE.PointsMaterial({
                    size: this.pointSize,
                    color: 0x60a5fa,
                    sizeAttenuation: true
                });
            }
            this.pointCloudObject = new THREE.Points(geometry, material);
            this.pointCloudObject.visible = this.showPointCloud;
            this.scene.add(this.pointCloudObject);
            this.updatePointStats(geometry);
        }, undefined, (err) => {
            console.debug("Point cloud load notice:", err);
        });
    }

    createCameraFrustumGeometry(scale = 0.12) {
        const w = scale * 0.7;
        const h = scale * 0.45;
        const z = scale * 1.0;

        const vertices = new Float32Array([
            0, 0, 0,  -w, -h, z,
            0, 0, 0,   w, -h, z,
            0, 0, 0,   w,  h, z,
            0, 0, 0,  -w,  h, z,
            -w, -h, z,   w, -h, z,
             w, -h, z,   w,  h, z,
             w,  h, z,  -w,  h, z,
            -w,  h, z,  -w, -h, z,
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
            const frustumGeom = this.createCameraFrustumGeometry(0.12);

            poses.forEach((pose, idx) => {
                const pos = new THREE.Vector3(pose.position[0], pose.position[1], pose.position[2]);
                points.push(pos);

                // Add frustum every 2 poses or endpoints
                if (idx % 2 === 0 || idx === poses.length - 1) {
                    let color = 0x3b82f6; // Blue intermediate
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

            const curveGeom = new THREE.BufferGeometry().setFromPoints(points);
            const lineMat = new THREE.LineBasicMaterial({ color: 0xec4899, linewidth: 2.5 });
            const line = new THREE.Line(curveGeom, lineMat);

            if (this.trajectoryObject) this.scene.remove(this.trajectoryObject);
            this.trajectoryObject = new THREE.Group();
            this.trajectoryObject.add(line);
            this.trajectoryObject.visible = this.showTrajectory;
            this.scene.add(this.trajectoryObject);

            this.frustumsGroup.visible = this.showFrustums;
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

            if (points.length === 0) return;

            const curveGeom = new THREE.BufferGeometry().setFromPoints(points);
            const lineMat = new THREE.LineBasicMaterial({ color: 0xec4899, linewidth: 2.5 });
            const line = new THREE.Line(curveGeom, lineMat);

            if (this.trajectoryObject) this.scene.remove(this.trajectoryObject);
            this.trajectoryObject = new THREE.Group();
            this.trajectoryObject.add(line);
            this.trajectoryObject.visible = this.showTrajectory;
            this.scene.add(this.trajectoryObject);

            this.updateTrajectoryStats(points.length);
        } catch (e) {
            console.warn("Failed to load trajectory TXT:", e);
        }
    }

    toggleMesh(visible) {
        this.showMesh = (visible !== undefined) ? visible : !this.showMesh;
        if (this.meshObject) this.meshObject.visible = this.showMesh;
    }

    togglePointCloud(visible) {
        this.showPointCloud = (visible !== undefined) ? visible : !this.showPointCloud;
        if (this.pointCloudObject) this.pointCloudObject.visible = this.showPointCloud;
    }

    toggleTrajectory(visible) {
        this.showTrajectory = (visible !== undefined) ? visible : !this.showTrajectory;
        if (this.trajectoryObject) this.trajectoryObject.visible = this.showTrajectory;
    }

    toggleFrustums(visible) {
        this.showFrustums = (visible !== undefined) ? visible : !this.showFrustums;
        if (this.frustumsGroup) this.frustumsGroup.visible = this.showFrustums;
    }

    toggleGrid(visible) {
        this.showGrid = (visible !== undefined) ? visible : !this.showGrid;
        if (this.gridHelper) this.gridHelper.visible = this.showGrid;
    }

    toggleAxes(visible) {
        this.showAxes = (visible !== undefined) ? visible : !this.showAxes;
        if (this.axesHelper) this.axesHelper.visible = this.showAxes;
    }

    setWireframe(enabled) {
        this.wireframeMode = (enabled !== undefined) ? enabled : !this.wireframeMode;
        if (this.meshObject) {
            if (this.meshObject.material) {
                this.meshObject.material.wireframe = this.wireframeMode;
            } else if (this.meshObject.children) {
                this.meshObject.traverse(child => {
                    if (child.isMesh && child.material) child.material.wireframe = this.wireframeMode;
                });
            }
        }
    }

    toggleWireframe() {
        this.setWireframe();
    }

    setPointSize(size) {
        this.pointSize = size;
        if (this.pointCloudObject && this.pointCloudObject.material) {
            this.pointCloudObject.material.size = size;
        }
    }

    resetView() {
        if (!this.camera || !this.controls) return;
        this.camera.position.set(0.0, 1.2, 2.4);
        this.controls.target.set(0, 0, 0);
        this.controls.update();
    }

    topView() {
        if (!this.camera || !this.controls) return;
        this.camera.position.set(0.0, 3.2, 0.001);
        this.controls.target.set(0, 0, 0);
        this.controls.update();
    }

    frontView() {
        if (!this.camera || !this.controls) return;
        this.camera.position.set(0.0, 0.2, 2.5);
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

"""Build go2_combined: the known-good go2_simplified.usd + the back-platform payload
rigidly merged into the `base` link.

Geometry comes from the original SolidWorks STL / COLLADA meshes, mass properties from
go2_back_platform.urdf, so nothing depends on the broken URDF->USD conversion.
"""

import os
import re
import struct
import itertools
import xml.etree.ElementTree as ET

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics, Vt

REPO = "/home/chenrypc/go2-rl-locomotion"
ASSETS = os.path.join(
    REPO, "source/go2_rl_locomotion/go2_rl_locomotion/tasks/manager_based/go2/assets"
)
GOOD_USD = os.path.join(ASSETS, "go2_simplified/go2_simplified.usd")
OUT_DIR = os.path.join(ASSETS, "go2_combined")
MESH_DIR = "/home/chenrypc/Desktop/urdf/back_platform/meshes"
URDF = "/home/chenrypc/Desktop/urdf/go2_description/urdf/go2_back_platform.urdf"

PAYLOAD_LINKS = ["jetson_link", "rail_plate", "rplidar_link", "lidar_link", "oak_link"]

# Comments carried into the generated .usda for the two spots where the URDF looks wrong.
NOTES = {
    "oak_link": (
        "\n                # NOTE: the URDF puts this 0.375 m ahead of the rail plate, i.e."
        "\n                # ~0.15 m in front of the robot's nose with nothing holding it up."
        "\n                # Looks like a typo for 0.0375; fix the URDF and re-run the script."
    ),
    "lidar_link": (
        "\n                # NOTE: the URDF's inertial block for this link (and for oak_link) is"
        "\n                # a copy of the other sensor's, with a COM ~0.19 m outside the part."
        "\n                # The mass properties below use the mesh instead."
    ),
}


# --------------------------------------------------------------------------------------
# mesh loading
# --------------------------------------------------------------------------------------
def load_stl(path):
    with open(path, "rb") as f:
        data = f.read()
    n = struct.unpack("<I", data[80:84])[0]
    arr = np.frombuffer(data[84 : 84 + 50 * n], dtype=np.uint8).reshape(n, 50)
    tri = arr[:, 12:48].copy().view("<f4").reshape(n, 3, 3).astype(np.float64)
    return weld(tri)


def load_dae(path):
    text = open(path, encoding="utf-8", errors="replace").read()
    verts = np.array(
        re.search(r'<float_array count="\d+" id="cubeverts-array0-array">(.*?)</float_array>', text, re.S)
        .group(1)
        .split(),
        dtype=np.float64,
    ).reshape(-1, 3)
    # <triangles> <p> holds interleaved (vertex, normal) indices
    p_txt = re.search(r"<triangles.*?<p>(.*?)</p>", text, re.S).group(1)
    idx = np.array(p_txt.split(), dtype=np.int64)
    n_inputs = len(re.findall(r"<input ", re.search(r"<triangles.*?</triangles>", text, re.S).group(0)))
    faces = idx.reshape(-1, n_inputs)[:, 0].reshape(-1, 3)
    return verts, faces


def weld(tri, decimals=7):
    """(n,3,3) triangle soup -> (points, faces) with shared vertices."""
    flat = tri.reshape(-1, 3)
    _, first, inv = np.unique(np.round(flat, decimals), axis=0, return_index=True, return_inverse=True)
    points = flat[first]
    return points, inv.reshape(-1, 3)


# --------------------------------------------------------------------------------------
# rigid body math
# --------------------------------------------------------------------------------------
def rpy_to_mat(rpy):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def mat_to_quat(R):
    """3x3 rotation matrix -> (w, x, y, z)."""
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x, y, z = (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x = (R[2, 1] - R[1, 2]) / s, 0.25 * s
        y, z = (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, y = (R[0, 2] - R[2, 0]) / s, 0.25 * s
        x, z = (R[0, 1] + R[1, 0]) / s, (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, z = (R[1, 0] - R[0, 1]) / s, 0.25 * s
        x, y = (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def quat_to_mat(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def mesh_inertia(points, faces):
    """Volume, COM and inertia tensor about the COM for a closed mesh of unit density."""
    v0, v1, v2 = points[faces[:, 0]], points[faces[:, 1]], points[faces[:, 2]]
    det = np.einsum("ij,ij->i", v0, np.cross(v1, v2))  # 6 * signed tet volume
    vol = det.sum() / 6.0
    com = (det[:, None] * (v0 + v1 + v2) / 4.0).sum(axis=0) / (6.0 * vol)

    # covariance of a tetrahedron (0, v0, v1, v2) w.r.t. the origin, unit density
    C = np.zeros((3, 3))
    for a in range(3):
        for b in range(3):
            s = (
                2 * (v0[:, a] * v0[:, b] + v1[:, a] * v1[:, b] + v2[:, a] * v2[:, b])
                + v0[:, a] * v1[:, b] + v1[:, a] * v0[:, b]
                + v0[:, a] * v2[:, b] + v2[:, a] * v0[:, b]
                + v1[:, a] * v2[:, b] + v2[:, a] * v1[:, b]
            )
            C[a, b] = (det * s / 120.0).sum()
    I_origin = np.trace(C) * np.eye(3) - C
    # shift to COM (unit density -> mass == vol)
    I_com = I_origin - vol * (com @ com * np.eye(3) - np.outer(com, com))
    return vol, com, I_com


def translate_inertia(I_com, mass, d):
    """Parallel axis: inertia about a point offset by d from the COM."""
    return I_com + mass * ((d @ d) * np.eye(3) - np.outer(d, d))


# --------------------------------------------------------------------------------------
# URDF payload description
# --------------------------------------------------------------------------------------
def parse_urdf():
    root = ET.parse(URDF).getroot()
    links, joints = {}, {}
    for l in root.findall("link"):
        name = l.get("name")
        if name not in PAYLOAD_LINKS:
            continue
        d = {"name": name}
        ine = l.find("inertial")
        o = ine.find("origin")
        d["mass"] = float(ine.find("mass").get("value"))
        d["com"] = np.array([float(v) for v in o.get("xyz").split()]) if o is not None and o.get("xyz") else np.zeros(3)
        i = ine.find("inertia")
        ixx, iyy, izz = float(i.get("ixx")), float(i.get("iyy")), float(i.get("izz"))
        ixy, ixz, iyz = float(i.get("ixy")), float(i.get("ixz")), float(i.get("iyz"))
        d["inertia"] = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
        vis = l.find("visual")
        vo = vis.find("origin")
        d["vis_xyz"] = np.array([float(v) for v in vo.get("xyz").split()]) if vo is not None and vo.get("xyz") else np.zeros(3)
        d["vis_rpy"] = np.array([float(v) for v in vo.get("rpy").split()]) if vo is not None and vo.get("rpy") else np.zeros(3)
        mesh = vis.find("geometry/mesh")
        d["mesh"] = os.path.join(MESH_DIR, os.path.basename(mesh.get("filename")))
        d["scale"] = np.array([float(v) for v in mesh.get("scale").split()]) if mesh.get("scale") else np.ones(3)
        links[name] = d
    for j in root.findall("joint"):
        child = j.find("child").get("link")
        if child not in PAYLOAD_LINKS:
            continue
        o = j.find("origin")
        joints[child] = {
            "parent": j.find("parent").get("link"),
            "xyz": np.array([float(v) for v in o.get("xyz").split()]),
            "rpy": np.array([float(v) for v in o.get("rpy").split()]),
        }
    return links, joints


def main():
    links, joints = parse_urdf()

    # ---- pose of every payload link in the `base` frame -------------------------------
    pose = {"base": (np.eye(3), np.zeros(3))}
    for name in PAYLOAD_LINKS:  # URDF order is already parent-first
        j = joints[name]
        Rp, tp = pose[j["parent"]]
        R = Rp @ rpy_to_mat(j["rpy"])
        t = tp + Rp @ j["xyz"]
        pose[name] = (R, t)

    print()
    for name in PAYLOAD_LINKS:
        R, t = pose[name]
        print(f"  {name:14s} origin in base = {np.round(t, 5)}  rpy_quat={np.round(mat_to_quat(R), 4)}")
    print()

    # ---- geometry + per-link mass properties -----------------------------------------
    geom = {}
    for name in PAYLOAD_LINKS:
        d = links[name]
        if d["mesh"].lower().endswith(".stl"):
            pts, faces = load_stl(d["mesh"])
        else:
            pts, faces = load_dae(d["mesh"])
        # bake the URDF visual origin + mesh scale into the points, so the USD prim
        # transform is exactly the link frame
        pts = pts * d["scale"]
        pts = pts @ rpy_to_mat(d["vis_rpy"]).T + d["vis_xyz"]
        vol, mcom, I_unit = mesh_inertia(pts, faces)
        geom[name] = (pts.astype(np.float32), faces.astype(np.int32))

        com, I = d["com"], d["inertia"]
        note = "urdf"
        # The URDF re-uses one copy-pasted inertial block for lidar_link and oak_link;
        # its COM sits ~19 cm away from either link origin, which is physically wrong.
        # Fall back to the mesh's own (uniform-density) properties for those two.
        if name in ("lidar_link", "oak_link"):
            com = mcom
            I = I_unit / vol * d["mass"]
            note = "mesh (urdf inertial is a copy-paste of the other sensor)"
        d["com_used"], d["inertia_used"], d["note"] = com, I, note
        print(
            f"{name:14s} m={d['mass']:8.5f} kg  com={np.round(com, 5)}  "
            f"mesh: {len(pts):6d} pts {len(faces):6d} tris  vol={vol * 1e6:8.1f} cm^3  [{note}]"
        )

    # ---- base link mass properties from the good USD ----------------------------------
    stage = Usd.Stage.Open(GOOD_USD)
    base = stage.GetPrimAtPath("/go2_simplified/base")
    m_api = UsdPhysics.MassAPI(base)
    m0 = float(m_api.GetMassAttr().Get())
    c0 = np.array(m_api.GetCenterOfMassAttr().Get())
    diag0 = np.array(m_api.GetDiagonalInertiaAttr().Get())
    q0 = m_api.GetPrincipalAxesAttr().Get()
    R0 = quat_to_mat(np.array([q0.GetReal(), *q0.GetImaginary()]))
    I0 = R0 @ np.diag(diag0) @ R0.T
    print(f"\nbase (go2_simplified) m={m0:.4f} com={np.round(c0, 5)} diag={np.round(diag0, 6)}")

    # ---- combine ----------------------------------------------------------------------
    total_m = m0
    total_mc = m0 * c0
    for name in PAYLOAD_LINKS:
        d = links[name]
        R, t = pose[name]
        total_m += d["mass"]
        total_mc += d["mass"] * (t + R @ d["com_used"])
    total_com = total_mc / total_m

    I_total = translate_inertia(I0, m0, c0 - total_com)
    for name in PAYLOAD_LINKS:
        d = links[name]
        R, t = pose[name]
        com_b = t + R @ d["com_used"]
        I_b = R @ d["inertia_used"] @ R.T
        I_total += translate_inertia(I_b, d["mass"], com_b - total_com)

    evals, evecs = np.linalg.eigh(I_total)
    # eigh returns the axes sorted by eigenvalue; re-order/flip them so the principal
    # frame stays as close to the base frame as possible (keeps the quaternion readable)
    perm = max(itertools.permutations(range(3)), key=lambda p: sum(abs(evecs[i, p[i]]) for i in range(3)))
    evals, evecs = evals[list(perm)], evecs[:, list(perm)]
    for i in range(3):
        if evecs[i, i] < 0:
            evecs[:, i] *= -1
    if np.linalg.det(evecs) < 0:
        evecs[:, 2] *= -1
    q_pa = mat_to_quat(evecs)
    assert np.allclose(evecs @ np.diag(evals) @ evecs.T, I_total, atol=1e-9)

    payload_m = total_m - m0
    print(f"\npayload total   m={payload_m:.4f} kg")
    print(f"combined base   m={total_m:.4f} kg  com={np.round(total_com, 6)}")
    print(f"                diagonalInertia={np.round(evals, 6)}")
    print(f"                principalAxes(w,x,y,z)={np.round(q_pa, 6)}")
    print("                inertia tensor:\n", np.round(I_total, 6))

    # ---- write the payload geometry layer ---------------------------------------------
    os.makedirs(OUT_DIR, exist_ok=True)
    geo_path = os.path.join(OUT_DIR, "payload_geometry.usd")
    gstage = Usd.Stage.CreateNew(geo_path) if not os.path.exists(geo_path) else Usd.Stage.Open(geo_path)
    gstage.GetRootLayer().Clear()
    UsdGeom.SetStageUpAxis(gstage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(gstage, 1.0)
    scope = UsdGeom.Scope.Define(gstage, "/payload")
    gstage.SetDefaultPrim(scope.GetPrim())
    colors = {
        "jetson_link": (0.15, 0.15, 0.17),
        "rail_plate": (0.79, 0.82, 0.93),
        "rplidar_link": (0.20, 0.20, 0.22),
        "lidar_link": (0.75, 0.75, 0.75),
        "oak_link": (0.55, 0.57, 0.60),
    }
    for name in PAYLOAD_LINKS:
        pts, faces = geom[name]
        # <name> is an Xform holding the mesh so the robot layer can reference it under a
        # `def Xform` and still mark the whole thing instanceable
        group = UsdGeom.Xform.Define(gstage, f"/payload/{name}")
        mesh = UsdGeom.Mesh.Define(gstage, f"/payload/{name}/geom")
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.reshape(-1)))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(faces), 3, dtype=np.int32)))
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.CreateDisplayColorAttr([Gf.Vec3f(*colors[name])])
        mesh.CreateExtentAttr(
            [Gf.Vec3f(*pts.min(axis=0).tolist()), Gf.Vec3f(*pts.max(axis=0).tolist())]
        )
        group.GetPrim().SetInstanceable(True)
    gstage.GetRootLayer().Save()
    print(f"\nwrote {geo_path} ({os.path.getsize(geo_path) / 1e6:.1f} MB)")

    # ---- write go2_combined.usda ------------------------------------------------------
    def fmt(v, n=9):
        return ", ".join(f"{0.0 if abs(x) < 1e-12 else round(float(x), n):g}" for x in v)

    body = []
    for name in PAYLOAD_LINKS:
        R, t = pose[name]
        q = mat_to_quat(R)
        parent = joints[name]["parent"]
        body.append(
            f'''
            def Xform "{name}" (
                prepend references = @./payload_geometry.usd@</payload/{name}>
            )
            {{
                # URDF joint {parent} -> {name} (fixed), resolved into the base frame{NOTES.get(name, "")}
                double3 xformOp:translate = ({fmt(t)})
                quatf xformOp:orient = ({fmt(q)})
                uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]
            }}
'''
        )

    usda = f'''#usda 1.0
(
    defaultPrim = "go2_combined"
    doc = """Unitree Go2 (simplified) with the back-platform payload rigidly merged into
the `base` link.

Kinematics/physics come straight from assets/go2_simplified/go2_simplified.usd -- the
articulation is untouched, so the body and joint sets are identical to the plain Go2.
The payload (jetson plate + rail plate + RPLidar + Livox lidar + OAK camera) is added as
visual-only geometry under /go2_combined/base/payload, and its mass is folded into the
base link's mass/COM/inertia below.

Payload mass properties come from go2_back_platform.urdf and the meshes from the
back_platform package -- nothing here depends on the URDF->USD conversion.

Regenerate with scripts/build_go2_combined.py."""
    kilogramsPerUnit = 1
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "go2_combined" (
    prepend references = @../go2_simplified/go2_simplified.usd@</go2_simplified>
)
{{
    over "base"
    {{
        # base ({m0:.4f} kg) + payload ({payload_m:.4f} kg), combined about the new COM
        float physics:mass = {total_m:.6f}
        point3f physics:centerOfMass = ({fmt(total_com)})
        float3 physics:diagonalInertia = ({fmt(evals)})
        quatf physics:principalAxes = ({fmt(q_pa)})

        # visual-only: the URDF declares no <collision> for any payload link
        def Xform "payload"
        {{
{"".join(body)}        }}
    }}
}}
'''
    out = os.path.join(OUT_DIR, "go2_combined.usda")
    with open(out, "w") as f:
        f.write(usda)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

"""
Parse an Assetto Corsa .kn5 file and extract all 3D vertex positions to CSV.
KN5 format reference: https://github.com/gro-ove/actools/wiki/KN5-file-format
"""

import struct
import csv
import sys
from pathlib import Path


def read_str(f):
    length = struct.unpack('<I', f.read(4))[0]
    return f.read(length).decode('utf-8', errors='replace')


def read_str_fixed(f, length):
    return f.read(length).decode('utf-8', errors='replace').rstrip('\x00')


def skip_texture(f, version):
    tex_type = struct.unpack('<I', f.read(4))[0]
    _name = read_str(f)
    if tex_type == 0:
        # active texture: has embedded data
        _active = struct.unpack('<I', f.read(4))[0]
        size = struct.unpack('<I', f.read(4))[0]
        f.seek(size, 1)
    else:
        # custom/external texture
        _size = struct.unpack('<I', f.read(2))[0]  # 2-byte size info
        # Actually for non-embedded: 4 bytes custom name follow
        # Re-read properly:
        f.seek(-2, 1)
        _active = struct.unpack('<I', f.read(4))[0]
        size = struct.unpack('<I', f.read(4))[0]
        f.seek(size, 1)


def skip_texture_v5(f):
    """KN5 v5+ texture block."""
    tex_type = struct.unpack('<i', f.read(4))[0]
    _name = read_str(f)
    if tex_type == 0:
        _active = struct.unpack('<i', f.read(4))[0]
        size = struct.unpack('<i', f.read(4))[0]
        f.seek(size, 1)
    else:
        _active = struct.unpack('<i', f.read(4))[0]
        size = struct.unpack('<i', f.read(4))[0]
        f.seek(size, 1)


def skip_material(f, version):
    _name = read_str(f)
    _shader = read_str(f)
    _alpha_blend = struct.unpack('<B', f.read(1))[0]
    _alpha_tested = struct.unpack('<B', f.read(1))[0]
    _alpha_ref = struct.unpack('<i', f.read(4))[0]
    _depth_mode = struct.unpack('<i', f.read(4))[0]
    prop_count = struct.unpack('<i', f.read(4))[0]
    for _ in range(prop_count):
        _prop_name = read_str(f)
        _value = struct.unpack('<f', f.read(4))[0]
        _vec2 = struct.unpack('<2f', f.read(8))
        _vec3 = struct.unpack('<3f', f.read(12))
        _vec4 = struct.unpack('<4f', f.read(16))
    tex_map_count = struct.unpack('<i', f.read(4))[0]
    for _ in range(tex_map_count):
        _slot = struct.unpack('<i', f.read(4))[0]
        _tex_name = read_str(f)


VERTEX_STRIDE = {
    # (positions_per_vertex, bytes_per_vertex)
    # Standard layout: pos(3f) + normal(3f) + uv(2f) + tangent(3f) = 44 bytes
}


def read_node(f, version, vertices_out):
    node_class = struct.unpack('<i', f.read(4))[0]
    _name = read_str(f)
    child_count = struct.unpack('<i', f.read(4))[0]
    _active = struct.unpack('<B', f.read(1))[0]

    if node_class == 1:
        # Transform node
        _matrix = struct.unpack('<16f', f.read(64))

    elif node_class == 2:
        # Mesh node
        _cast_shadows = struct.unpack('<B', f.read(1))[0]
        _visible = struct.unpack('<B', f.read(1))[0]
        _transparent = struct.unpack('<B', f.read(1))[0]

        vert_count = struct.unpack('<I', f.read(4))[0]
        for _ in range(vert_count):
            x, y, z = struct.unpack('<3f', f.read(12))
            vertices_out.append((x, y, z))
            # skip normal(3f) + uv(2f) + tangent(3f) = 8 floats = 32 bytes
            f.seek(32, 1)

        idx_count = struct.unpack('<I', f.read(4))[0]
        f.seek(idx_count * 2, 1)  # uint16 indices

        _mat_id = struct.unpack('<i', f.read(4))[0]
        _layer = struct.unpack('<I', f.read(4))[0]
        _lod_in = struct.unpack('<f', f.read(4))[0]
        _lod_out = struct.unpack('<f', f.read(4))[0]
        _bb_min = struct.unpack('<3f', f.read(12))
        _bb_max = struct.unpack('<3f', f.read(12))
        _renderable = struct.unpack('<i', f.read(4))[0]

    elif node_class == 3:
        # Skinned mesh node
        _cast_shadows = struct.unpack('<B', f.read(1))[0]
        _visible = struct.unpack('<B', f.read(1))[0]
        _transparent = struct.unpack('<B', f.read(1))[0]

        bone_count = struct.unpack('<I', f.read(4))[0]
        for _ in range(bone_count):
            _bone_name = read_str(f)
            _bone_matrix = struct.unpack('<16f', f.read(64))

        vert_count = struct.unpack('<I', f.read(4))[0]
        for _ in range(vert_count):
            x, y, z = struct.unpack('<3f', f.read(12))
            vertices_out.append((x, y, z))
            # normal(3f) + uv(2f) + tangent(3f) + bone_weights(4f) + bone_indices(4B)
            f.seek(12 + 8 + 12 + 16 + 4, 1)

        idx_count = struct.unpack('<I', f.read(4))[0]
        f.seek(idx_count * 2, 1)

        _mat_id = struct.unpack('<i', f.read(4))[0]
        _layer = struct.unpack('<I', f.read(4))[0]
        _lod_in = struct.unpack('<f', f.read(4))[0]
        _lod_out = struct.unpack('<f', f.read(4))[0]
        _bb_min = struct.unpack('<3f', f.read(12))
        _bb_max = struct.unpack('<3f', f.read(12))
        _renderable = struct.unpack('<i', f.read(4))[0]

    for _ in range(child_count):
        read_node(f, version, vertices_out)


def parse_kn5(path):
    vertices = []
    with open(path, 'rb') as f:
        magic = f.read(6)
        if magic != b'sc6969':
            raise ValueError(f"Not a KN5 file (magic: {magic!r})")

        version = struct.unpack('<I', f.read(4))[0]
        print(f"KN5 version: {version}")

        if version > 5:
            _unknown = struct.unpack('<I', f.read(4))[0]

        tex_count = struct.unpack('<I', f.read(4))[0]
        print(f"Textures: {tex_count}")
        for i in range(tex_count):
            skip_texture_v5(f)
            if (i + 1) % 10 == 0:
                print(f"  Skipped {i+1}/{tex_count} textures...")

        mat_count = struct.unpack('<I', f.read(4))[0]
        print(f"Materials: {mat_count}")
        for _ in range(mat_count):
            skip_material(f, version)

        print("Parsing node tree...")
        read_node(f, version, vertices)

    return vertices


def main():
    kn5_path = Path(r"e:\Users\Edward\Documents\Code\LHR_code\path_gen\autox\Michigan-2018-Assetto-Import\fsae_michigan18.kn5")
    out_path = Path(r"e:\Users\Edward\Documents\Code\LHR_code\path_gen\autox\Michigan-2018-Assetto-Import\michigan18_vertices.csv")

    print(f"Parsing {kn5_path.name} ({kn5_path.stat().st_size / 1e6:.1f} MB)...")
    vertices = parse_kn5(kn5_path)
    print(f"Extracted {len(vertices):,} vertices")

    print(f"Writing to {out_path.name}...")
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['x', 'y', 'z'])
        writer.writerows(vertices)

    print("Done.")


if __name__ == '__main__':
    main()

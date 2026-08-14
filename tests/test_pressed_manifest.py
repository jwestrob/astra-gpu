import copy
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import types
import unittest
from unittest import mock
from pathlib import Path

import pyhmmer


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = ROOT / "python" / "plan7_gpu"
HMM_FIXTURE = ROOT / "refs" / "src" / "hmmer-3.4" / "tutorial" / "globins4.hmm"


def _load_manifest_module():
    package_name = "_plan7_gpu_manifest_test"
    package = types.ModuleType(package_name)
    package.__path__ = [str(PACKAGE_DIR)]
    sys.modules[package_name] = package
    name = f"{package_name}.pressed_manifest"
    spec = importlib.util.spec_from_file_location(
        name, PACKAGE_DIR / "pressed_manifest.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load pressed-manifest implementation")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


pressed_manifest = _load_manifest_module()


class PressedManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(
            prefix="plan7-pressed-manifest-test-"
        )
        cls.directory = Path(cls.temporary.name)
        cls.base = cls.directory / "source" / "globins"
        cls.base.parent.mkdir()
        with pyhmmer.plan7.HMMFile(HMM_FIXTURE) as hmm_file:
            cls.hmm = next(hmm_file)
        pyhmmer.hmmer.hmmpress([cls.hmm], cls.base)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def setUp(self):
        self.manifest_path = self.directory / self.id().split(".")[-1] / "globins.json"

    def test_full_audit_records_hashes_and_supports_safe_fast_path(self):
        manifest = pressed_manifest.create_pressed_manifest(
            self.base, self.manifest_path
        )
        self.assertEqual(manifest["schema"], "plan7-gpu-pressed-profile-manifest")
        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(manifest["database"]["model_count"], 1)
        self.assertEqual(manifest["verification"]["models_compared"], 1)
        self.assertEqual(
            set(manifest["database"]["artifacts"]),
            {"h3m", "h3i", "h3f", "h3p"},
        )
        for suffix, artifact in manifest["database"]["artifacts"].items():
            member = Path(f"{self.base}.{suffix}")
            self.assertEqual(
                artifact["sha256"], hashlib.sha256(member.read_bytes()).hexdigest()
            )
            self.assertEqual(artifact["size_bytes"], member.stat().st_size)

        validation = pressed_manifest.validate_pressed_manifest(
            self.base, self.manifest_path
        )
        self.assertEqual(validation.model_count, 1)
        self.assertFalse(validation.used_content_hashes)
        self.assertEqual(
            validation.manifest_sha256,
            hashlib.sha256(self.manifest_path.read_bytes()).hexdigest(),
        )

    def test_identical_copy_uses_hash_fallback_and_corruption_is_rejected(self):
        pressed_manifest.create_pressed_manifest(self.base, self.manifest_path)
        copied_base = self.directory / self.id().split(".")[-1] / "copy" / "globins"
        copied_base.parent.mkdir()
        for suffix in pressed_manifest.PRESSED_SUFFIXES:
            shutil.copy2(f"{self.base}.{suffix}", f"{copied_base}.{suffix}")

        validation = pressed_manifest.validate_pressed_manifest(
            copied_base, self.manifest_path
        )
        self.assertTrue(validation.used_content_hashes)
        with self.assertRaisesRegex(
            pressed_manifest.PressedManifestError, "stat token"
        ):
            pressed_manifest.validate_pressed_manifest(
                copied_base,
                self.manifest_path,
                allow_hash_fallback=False,
            )

        filter_path = Path(f"{copied_base}.h3f")
        with filter_path.open("r+b") as stream:
            original = stream.read(1)
            stream.seek(0)
            stream.write(bytes([original[0] ^ 1]))
        with self.assertRaisesRegex(
            pressed_manifest.PressedManifestError, "SHA256 mismatch for h3f"
        ):
            pressed_manifest.validate_pressed_manifest(copied_base, self.manifest_path)

    def test_runtime_private_abi_is_bound(self):
        pressed_manifest.create_pressed_manifest(self.base, self.manifest_path)
        manifest = json.loads(self.manifest_path.read_text(encoding="ascii"))
        manifest["tool"]["private_abi"]["sha256"] = "0" * 64
        self.manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="ascii",
        )
        with self.assertRaisesRegex(
            pressed_manifest.PressedManifestError, "private ABI"
        ):
            pressed_manifest.validate_pressed_manifest(self.base, self.manifest_path)

    def test_audit_source_is_bound(self):
        pressed_manifest.create_pressed_manifest(self.base, self.manifest_path)
        manifest = json.loads(self.manifest_path.read_text(encoding="ascii"))
        self.assertEqual(manifest["tool"]["source"]["filename"], "pressed_manifest.py")
        manifest["tool"]["source"]["sha256"] = "0" * 64
        self.manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="ascii",
        )
        with self.assertRaisesRegex(
            pressed_manifest.PressedManifestError, "audit source"
        ):
            pressed_manifest.validate_pressed_manifest(self.base, self.manifest_path)

    def test_source_replaced_after_import_cannot_mint_safe_source_claim(self):
        directory = self.directory / self.id().split(".")[-1]
        package_directory = directory / "package"
        package_directory.mkdir(parents=True)
        source_path = package_directory / "pressed_manifest.py"
        safe_source = (PACKAGE_DIR / "pressed_manifest.py").read_text(encoding="utf-8")
        weakened_source = safe_source.replace(
            "model_count = _audit_profiles(snapshot)",
            "model_count = 1  # deliberately weakened test copy",
        )
        self.assertNotEqual(weakened_source, safe_source)
        source_path.write_text(weakened_source, encoding="utf-8")
        shutil.copyfile(PACKAGE_DIR / "_abi.py", package_directory / "_abi.py")

        package_name = "_plan7_gpu_manifest_replaced_source_test"
        package = types.ModuleType(package_name)
        package.__path__ = [str(package_directory)]
        sys.modules[package_name] = package
        module_name = f"{package_name}.pressed_manifest"
        spec = importlib.util.spec_from_file_location(module_name, source_path)
        if spec is None or spec.loader is None:
            self.fail("cannot load isolated pressed-manifest implementation")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        source_path.write_text(safe_source, encoding="utf-8")
        with self.assertRaisesRegex(
            module.PressedManifestError, "changed after it was imported"
        ):
            module.build_pressed_manifest(self.base)

    def test_manifest_cannot_overwrite_database_base(self):
        directory = self.directory / self.id().split(".")[-1]
        directory.mkdir()
        base = directory / "HydDB_all_MM2022.hmm"
        source = HMM_FIXTURE.read_bytes()
        base.write_bytes(source)
        pyhmmer.hmmer.hmmpress([self.hmm], base)

        with self.assertRaisesRegex(
            pressed_manifest.PressedManifestError, "database base"
        ):
            pressed_manifest.create_pressed_manifest(base, base, overwrite=True)
        self.assertEqual(base.read_bytes(), source)

    def test_schema_rejects_omitted_mutated_and_extra_fields(self):
        manifest = pressed_manifest.create_pressed_manifest(
            self.base, self.manifest_path
        )

        def remove_created_at(value):
            value.pop("created_at_utc")

        def mutate_created_at(value):
            value["created_at_utc"] = "2026-08-14T12:31:57+00:00"

        def mutate_reference(value):
            value["verification"]["reference"] = "a different audit"

        def mutate_comparison(value):
            value["verification"]["comparison"] = "a different comparison"

        def add_root_field(value):
            value["unexpected"] = True

        def add_tool_field(value):
            value["tool"]["unexpected"] = True

        def add_database_field(value):
            value["database"]["unexpected"] = True

        def add_artifact_field(value):
            value["database"]["artifacts"]["h3m"]["unexpected"] = True

        def add_stat_field(value):
            value["database"]["stat_token"]["h3m"]["unexpected"] = True

        def add_verification_field(value):
            value["verification"]["unexpected"] = True

        cases = {
            "omitted created_at_utc": remove_created_at,
            "noncanonical created_at_utc": mutate_created_at,
            "mutated reference": mutate_reference,
            "mutated comparison": mutate_comparison,
            "extra root field": add_root_field,
            "extra tool field": add_tool_field,
            "extra database field": add_database_field,
            "extra artifact field": add_artifact_field,
            "extra stat field": add_stat_field,
            "extra verification field": add_verification_field,
        }
        directory = self.manifest_path.parent
        for label, mutate in cases.items():
            with self.subTest(label):
                candidate = copy.deepcopy(manifest)
                mutate(candidate)
                path = directory / f"{label.replace(' ', '-')}.json"
                path.write_text(
                    json.dumps(candidate, indent=2, sort_keys=True) + "\n",
                    encoding="ascii",
                )
                with self.assertRaises(pressed_manifest.PressedManifestError):
                    pressed_manifest.validate_pressed_manifest(self.base, path)

    def test_pinned_view_rejects_in_place_member_mutation(self):
        directory = self.directory / self.id().split(".")[-1]
        directory.mkdir()
        base = directory / "globins"
        for suffix in pressed_manifest.PRESSED_SUFFIXES:
            shutil.copyfile(f"{self.base}.{suffix}", f"{base}.{suffix}")

        with self.assertRaisesRegex(
            pressed_manifest.PressedManifestError, "members were pinned"
        ):
            with pressed_manifest._pinned_pressed_database(base):
                member = Path(f"{base}.h3f")
                metadata = member.stat()
                with member.open("r+b") as stream:
                    first = stream.read(1)
                    stream.seek(0)
                    stream.write(bytes([first[0] ^ 1]))
                os.utime(
                    member,
                    ns=(metadata.st_atime_ns, metadata.st_mtime_ns + 1_000_000_000),
                )

    def test_pinned_audit_cannot_be_redirected_by_parent_symlink_swap(self):
        original = self.hmm.copy()
        perturbed = original.copy()
        row = perturbed.match_emissions[1]
        source, destination = sorted(range(len(row)), key=row.__getitem__)[:2]
        epsilon = min(float(row[source]) / 2.0, 1e-4)
        row[source] -= epsilon
        row[destination] += epsilon
        perturbed.renormalize()
        self.assertEqual(perturbed.consensus, original.consensus)

        root = self.directory / self.id().split(".")[-1]
        victim_directory = root / "victim"
        valid_directory = root / "valid"
        victim_directory.mkdir(parents=True)
        valid_directory.mkdir()
        victim_base = victim_directory / "database"
        valid_base = valid_directory / "database"
        original_base = root / "original"
        perturbed_base = root / "perturbed"
        pyhmmer.hmmer.hmmpress([original], original_base)
        pyhmmer.hmmer.hmmpress([perturbed], perturbed_base)
        pyhmmer.hmmer.hmmpress([original], valid_base)
        for suffix in ("h3m", "h3i"):
            shutil.copyfile(f"{original_base}.{suffix}", f"{victim_base}.{suffix}")
        for suffix in ("h3f", "h3p"):
            shutil.copyfile(f"{perturbed_base}.{suffix}", f"{victim_base}.{suffix}")

        audit_profiles = pressed_manifest._audit_profiles
        parked_directory = root / "parked"

        def swap_then_audit(pinned_base):
            os.rename(victim_directory, parked_directory)
            os.symlink(valid_directory, victim_directory, target_is_directory=True)
            try:
                return audit_profiles(pinned_base)
            finally:
                victim_directory.unlink()
                os.rename(parked_directory, victim_directory)

        with (
            mock.patch.object(
                pressed_manifest,
                "_audit_profiles",
                side_effect=swap_then_audit,
            ),
            self.assertRaisesRegex(
                pressed_manifest.PressedProfileAuditError,
                "score mismatch.*ordinal 0",
            ),
        ):
            pressed_manifest.build_pressed_manifest(victim_base)

    def test_mixed_hmm_and_profile_artifacts_fail_the_full_audit(self):
        original = self.hmm.copy()
        perturbed = original.copy()
        row = perturbed.match_emissions[1]
        source, destination = sorted(range(len(row)), key=row.__getitem__)[:2]
        epsilon = min(float(row[source]) / 2.0, 1e-4)
        row[source] -= epsilon
        row[destination] += epsilon
        perturbed.renormalize()
        self.assertEqual(perturbed.consensus, original.consensus)

        directory = self.directory / self.id().split(".")[-1]
        directory.mkdir()
        original_base = directory / "original"
        perturbed_base = directory / "perturbed"
        mixed_base = directory / "mixed"
        pyhmmer.hmmer.hmmpress([original], original_base)
        pyhmmer.hmmer.hmmpress([perturbed], perturbed_base)
        for suffix in ("h3m", "h3i"):
            shutil.copyfile(f"{original_base}.{suffix}", f"{mixed_base}.{suffix}")
        for suffix in ("h3f", "h3p"):
            shutil.copyfile(f"{perturbed_base}.{suffix}", f"{mixed_base}.{suffix}")

        with self.assertRaisesRegex(
            pressed_manifest.PressedProfileAuditError,
            "score mismatch.*ordinal 0",
        ):
            pressed_manifest.build_pressed_manifest(mixed_base)

    def test_mixed_index_offsets_fail_the_full_audit(self):
        first = self.hmm.copy()
        first.name = "z-model"
        second = self.hmm.copy()
        second.name = "a-model"

        directory = self.directory / self.id().split(".")[-1]
        directory.mkdir()
        forward_base = directory / "forward"
        reverse_base = directory / "reverse"
        mixed_base = directory / "mixed"
        pyhmmer.hmmer.hmmpress([first, second], forward_base)
        pyhmmer.hmmer.hmmpress([second, first], reverse_base)
        for suffix in ("h3m", "h3f", "h3p"):
            shutil.copyfile(f"{forward_base}.{suffix}", f"{mixed_base}.{suffix}")
        shutil.copyfile(f"{reverse_base}.h3i", f"{mixed_base}.h3i")

        with self.assertRaisesRegex(
            pressed_manifest.PressedProfileAuditError,
            "index offset/alias mismatch.*ordinal 0",
        ):
            pressed_manifest.build_pressed_manifest(mixed_base)


if __name__ == "__main__":
    unittest.main()

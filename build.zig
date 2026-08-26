const std = @import("std");

const opencv_libs = [_][]const u8{
    "opencv_core",
    "opencv_imgproc",
    "opencv_imgcodecs",
    "opencv_video",
    "opencv_videoio",
    "opencv_objdetect",
    "opencv_dnn",
};

const gtest_libs = [_][]const u8{
    "gmock_main",
    "gmock",
    "gtest",
};

const cxx_flags = [_][]const u8{
    "-std=c++17",
    "-Wall",
    "-Wextra",
};

const header_files = [_][]const u8{
    "include/vidicant/vidicant.hpp",
    "include/vidicant/c_api.h",
    "include/vidicant/types.hpp",
    "include/vidicant/pipeline.hpp",
    "include/vidicant/controller.hpp",
    "include/vidicant/image.hpp",
    "include/vidicant/video.hpp",
    "include/vidicant/core/dedupe.hpp",
    "include/vidicant/core/image_ops.hpp",
    "include/vidicant/core/video_ops.hpp",
    "include/vidicant/dnn/dnn_engine.hpp",
    "include/vidicant/io/file_detector.hpp",
    "include/vidicant/cli/dedupe_cmd.hpp",
    "include/vidicant/cli/filter.hpp",
    "include/vidicant/cli/formatters.hpp",
};

const lib_sources = [_][]const u8{
    "src/core/dedupe.cpp",
    "src/core/image_ops.cpp",
    "src/core/video_ops.cpp",
    "src/dnn/dnn_engine.cpp",
    "src/io/file_detector.cpp",
    "src/image.cpp",
    "src/video.cpp",
    "src/pipeline.cpp",
    "src/vidicant_c_api.cpp",
};

const cli_sources = [_][]const u8{
    "src/core/dedupe.cpp",
    "src/core/image_ops.cpp",
    "src/core/video_ops.cpp",
    "src/dnn/dnn_engine.cpp",
    "src/io/file_detector.cpp",
    "src/main.cpp",
    "src/pipeline.cpp",
    "src/image.cpp",
    "src/video.cpp",
};

fn addIncludeIfExists(b: *std.Build, mod: *std.Build.Module, path: []const u8) void {
    if (std.Io.Dir.accessAbsolute(b.graph.io, path, .{})) |_| {
        mod.addIncludePath(.{ .cwd_relative = path });
    } else |_| {}
}

fn addLibraryIfExists(b: *std.Build, mod: *std.Build.Module, path: []const u8) void {
    if (std.Io.Dir.accessAbsolute(b.graph.io, path, .{})) |_| {
        mod.addLibraryPath(.{ .cwd_relative = path });
    } else |_| {}
}

fn configureDependencies(b: *std.Build, mod: *std.Build.Module, target: std.Build.ResolvedTarget, custom_path: ?[]const u8) void {
    if (custom_path) |p| {
        const inc_path = b.fmt("{s}/include", .{p});
        const lib_path = b.fmt("{s}/lib", .{p});
        addIncludeIfExists(b, mod, inc_path);
        addLibraryIfExists(b, mod, lib_path);
        return;
    }

    const os = target.result.os.tag;
    if (os == .macos) {
        const candidate_includes = [_][]const u8{
            "/opt/homebrew/include",
            "/opt/homebrew/opt/opencv/include/opencv5",
            "/opt/homebrew/opt/opencv/include/opencv4",
            "/opt/homebrew/opt/googletest/include",
            "/usr/local/include",
            "/usr/local/opt/opencv/include/opencv5",
            "/usr/local/opt/opencv/include/opencv4",
            "/usr/local/opt/googletest/include",
        };
        for (candidate_includes) |inc| {
            addIncludeIfExists(b, mod, inc);
        }

        const candidate_libs = [_][]const u8{
            "/opt/homebrew/lib",
            "/opt/homebrew/opt/opencv/lib",
            "/opt/homebrew/opt/googletest/lib",
            "/usr/local/lib",
            "/usr/local/opt/opencv/lib",
            "/usr/local/opt/googletest/lib",
        };
        for (candidate_libs) |lib_path| {
            addLibraryIfExists(b, mod, lib_path);
        }
    }
}

fn createVidicantModule(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    sources: []const []const u8,
    opencv_path: ?[]const u8,
    link_gtest: bool,
) *std.Build.Module {
    const mod = b.createModule(.{
        .target = target,
        .optimize = optimize,
    });
    mod.addCSourceFiles(.{
        .files = sources,
        .flags = &cxx_flags,
    });
    mod.addIncludePath(.{ .cwd_relative = "include" });
    configureDependencies(b, mod, target, opencv_path);
    for (opencv_libs) |lib_name| {
        mod.linkSystemLibrary(lib_name, .{});
    }
    if (link_gtest) {
        for (gtest_libs) |lib_name| {
            mod.linkSystemLibrary(lib_name, .{});
        }
    }
    mod.link_libc = true;
    mod.link_libcpp = true;
    return mod;
}

fn createLinuxCxxCommand(
    b: *std.Build,
    sources: []const []const u8,
    opencv_path: ?[]const u8,
    output_name: []const u8,
    is_shared_lib: bool,
    link_gtest: bool,
) std.Build.LazyPath {
    const cmd = b.addSystemCommand(&.{ "c++", "-std=c++17", "-O3", "-Wno-psabi", "-I", "include" });
    if (is_shared_lib) {
        cmd.addArgs(&.{ "-fPIC", "-shared" });
    }
    if (opencv_path) |p| {
        cmd.addArgs(&.{ "-I", b.fmt("{s}/include", .{p}), "-L", b.fmt("{s}/lib", .{p}) });
    } else {
        cmd.addArgs(&.{ "-I", "/usr/include/opencv4", "-I", "/usr/include/opencv5", "-I", "/usr/local/include", "-I", "/usr/include" });
    }
    for (header_files) |hdr| {
        cmd.addFileInput(b.path(hdr));
    }
    for (sources) |src| {
        cmd.addFileArg(b.path(src));
    }
    for (opencv_libs) |lib_name| {
        cmd.addArg(b.fmt("-l{s}", .{lib_name}));
    }
    if (link_gtest) {
        for (gtest_libs) |lib_name| {
            cmd.addArg(b.fmt("-l{s}", .{lib_name}));
        }
        cmd.addArg("-lpthread");
    }
    cmd.addArg("-o");
    return cmd.addOutputFileArg(output_name);
}

fn stageLibraryToPkg(b: *std.Build, lib_file: std.Build.LazyPath) void {
    const copy_pkg = b.addSystemCommand(&.{"cp"});
    copy_pkg.addFileArg(lib_file);
    copy_pkg.addArg("vidicant/");
    b.getInstallStep().dependOn(&copy_pkg.step);
}

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const opencv_path = b.option([]const u8, "opencv-path", "Custom path to OpenCV installation directory");
    const install_to_pkg = b.option(bool, "install-to-pkg", "Copy shared library directly into vidicant/ directory for wheel staging") orelse false;
    const os = target.result.os.tag;

    // Test steps
    const test_step = b.step("test", "Run all unit and end-to-end tests");
    const test_native_step = b.step("test-native", "Run native C++ unit tests (GTest/GMock)");
    const test_e2e_step = b.step("test-e2e", "Run Python end-to-end test suite");

    const test_e2e_cmd = b.addSystemCommand(&.{ "python3", "e2e.py" });
    test_e2e_cmd.setEnvironmentVariable("PYTHONPATH", ".");
    test_e2e_cmd.step.dependOn(b.getInstallStep());
    test_e2e_step.dependOn(&test_e2e_cmd.step);

    test_step.dependOn(test_native_step);
    test_step.dependOn(test_e2e_step);

    // Auto-discover test files in test/ directory
    var test_files: std.ArrayList([]const u8) = .empty;
    const io = b.graph.io;
    var test_dir = b.build_root.handle.openDir(io, "test", .{ .iterate = true }) catch null;
    if (test_dir) |*dir| {
        defer dir.close(io);
        var iter = dir.iterate();
        while (iter.next(io) catch null) |entry| {
            if (entry.kind == .file and std.mem.startsWith(u8, entry.name, "test_") and std.mem.endsWith(u8, entry.name, ".cpp")) {
                const path = b.fmt("test/{s}", .{entry.name});
                test_files.append(b.allocator, path) catch @panic("OOM");
            }
        }
    }
    // Sort for deterministic test execution order
    std.mem.sort([]const u8, test_files.items, {}, struct {
        fn lessThan(_: void, a: []const u8, b_str: []const u8) bool {
            return std.mem.lessThan(u8, a, b_str);
        }
    }.lessThan);

    // Linux: System OpenCV packages use GNU libstdc++ ABI.
    // Use host C++ compiler to guarantee ABI compatibility with system OpenCV.
    if (os == .linux) {
        // Build shared library (libvidicant.so)
        const lib_out = createLinuxCxxCommand(b, &lib_sources, opencv_path, "libvidicant.so", true, false);
        const install_lib = b.addInstallFile(lib_out, "lib/libvidicant.so");
        b.getInstallStep().dependOn(&install_lib.step);

        if (install_to_pkg) {
            stageLibraryToPkg(b, lib_out);
        }

        // Build CLI executable (vidicant_cli)
        const exe_out = createLinuxCxxCommand(b, &cli_sources, opencv_path, "vidicant_cli", false, false);
        const install_exe = b.addInstallFile(exe_out, "bin/vidicant_cli");
        b.getInstallStep().dependOn(&install_exe.step);

        // Native unit tests on Linux (auto-discovered)
        for (test_files.items) |test_file| {
            const test_basename = std.fs.path.stem(test_file);
            const test_sources = b.allocator.alloc([]const u8, lib_sources.len + 1) catch @panic("OOM");
            @memcpy(test_sources[0..lib_sources.len], &lib_sources);
            test_sources[lib_sources.len] = test_file;

            const test_exe = createLinuxCxxCommand(b, test_sources, opencv_path, test_basename, false, true);
            const run_test = std.Build.Step.Run.create(b, b.fmt("run {s}", .{test_basename}));
            run_test.addFileArg(test_exe);
            test_native_step.dependOn(&run_test.step);
        }
        return;
    }

    // macOS: Use Zig's native Clang toolchain + libc++
    const lib_mod = createVidicantModule(b, target, optimize, &lib_sources, opencv_path, false);
    const lib = b.addLibrary(.{
        .name = "vidicant",
        .linkage = .dynamic,
        .root_module = lib_mod,
    });
    b.installArtifact(lib);

    if (install_to_pkg) {
        stageLibraryToPkg(b, lib.getEmittedBin());
    }

    const exe_mod = createVidicantModule(b, target, optimize, &cli_sources, opencv_path, false);
    const exe = b.addExecutable(.{
        .name = "vidicant_cli",
        .root_module = exe_mod,
    });
    b.installArtifact(exe);

    // Native C++ tests (GTest) - auto-discovered
    for (test_files.items) |test_file| {
        const test_basename = std.fs.path.stem(test_file);
        const test_sources = b.allocator.alloc([]const u8, lib_sources.len + 1) catch @panic("OOM");
        @memcpy(test_sources[0..lib_sources.len], &lib_sources);
        test_sources[lib_sources.len] = test_file;

        const test_mod = createVidicantModule(b, target, optimize, test_sources, opencv_path, true);
        const test_exe = b.addExecutable(.{
            .name = test_basename,
            .root_module = test_mod,
        });
        const run_test = b.addRunArtifact(test_exe);
        test_native_step.dependOn(&run_test.step);
    }
}

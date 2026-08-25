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

const header_files = [_][]const u8{
    "include/vidicant/c_api.h",
    "include/vidicant/controller.hpp",
    "include/vidicant/image.hpp",
    "include/vidicant/video.hpp",
};

const lib_sources = [_][]const u8{
    "src/image.cpp",
    "src/video.cpp",
    "src/controller.cpp",
    "src/vidicant_c_api.cpp",
};

const cli_sources = [_][]const u8{
    "src/main.cpp",
    "src/controller.cpp",
    "src/image.cpp",
    "src/video.cpp",
};

fn configureOpenCV(b: *std.Build, mod: *std.Build.Module, target: std.Build.ResolvedTarget, custom_path: ?[]const u8) void {
    if (custom_path) |p| {
        const inc_path = b.fmt("{s}/include", .{p});
        const lib_path = b.fmt("{s}/lib", .{p});
        mod.addIncludePath(.{ .cwd_relative = inc_path });
        mod.addLibraryPath(.{ .cwd_relative = lib_path });
        return;
    }

    const os = target.result.os.tag;
    if (os == .macos) {
        mod.addIncludePath(.{ .cwd_relative = "/opt/homebrew/include" });
        mod.addIncludePath(.{ .cwd_relative = "/opt/homebrew/opt/opencv/include/opencv5" });
        mod.addIncludePath(.{ .cwd_relative = "/opt/homebrew/opt/opencv/include/opencv4" });
        mod.addLibraryPath(.{ .cwd_relative = "/opt/homebrew/opt/opencv/lib" });
    } else if (os == .windows) {
        mod.addIncludePath(.{ .cwd_relative = "C:/opencv/build/include" });
        mod.addLibraryPath(.{ .cwd_relative = "C:/opencv/build/x64/vc16/lib" });
    }
}

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const opencv_path = b.option([]const u8, "opencv-path", "Custom path to OpenCV installation directory");
    const install_to_pkg = b.option(bool, "install-to-pkg", "Copy shared library directly into vidicant/ directory for wheel staging") orelse false;
    const os = target.result.os.tag;

    // Test step: Runs the python e2e test suite
    const test_step = b.step("test", "Run end-to-end test suite");
    const test_cmd = b.addSystemCommand(&.{ "python3", "e2e.py" });
    test_cmd.setEnvironmentVariable("PYTHONPATH", ".");
    test_cmd.step.dependOn(b.getInstallStep());
    test_step.dependOn(&test_cmd.step);

    // Linux: System OpenCV packages use GNU libstdc++ ABI.
    // Use host C++ compiler to guarantee ABI compatibility with system OpenCV.
    if (os == .linux) {
        // Build shared library (libvidicant.so)
        const lib_cmd = b.addSystemCommand(&.{ "c++", "-std=c++17", "-O3", "-fPIC", "-shared", "-Wno-psabi", "-I", "include" });
        if (opencv_path) |p| {
            lib_cmd.addArgs(&.{ "-I", b.fmt("{s}/include", .{p}), "-L", b.fmt("{s}/lib", .{p}) });
        } else {
            lib_cmd.addArgs(&.{ "-I", "/usr/include/opencv4", "-I", "/usr/include/opencv5", "-I", "/usr/local/include", "-I", "/usr/include" });
        }
        for (header_files) |hdr| {
            lib_cmd.addFileInput(b.path(hdr));
        }
        for (lib_sources) |src| {
            lib_cmd.addFileArg(b.path(src));
        }
        for (opencv_libs) |lib_name| {
            lib_cmd.addArg(b.fmt("-l{s}", .{lib_name}));
        }
        lib_cmd.addArg("-o");
        const lib_out = lib_cmd.addOutputFileArg("libvidicant.so");
        const install_lib = b.addInstallFile(lib_out, "lib/libvidicant.so");
        b.getInstallStep().dependOn(&install_lib.step);

        if (install_to_pkg) {
            const copy_pkg = b.addSystemCommand(&.{"cp"});
            copy_pkg.addFileArg(lib_out);
            copy_pkg.addArg("vidicant/");
            b.getInstallStep().dependOn(&copy_pkg.step);
        }

        // Build CLI executable (vidicant_cli)
        const exe_cmd = b.addSystemCommand(&.{ "c++", "-std=c++17", "-O3", "-Wno-psabi", "-I", "include" });
        if (opencv_path) |p| {
            exe_cmd.addArgs(&.{ "-I", b.fmt("{s}/include", .{p}), "-L", b.fmt("{s}/lib", .{p}) });
        } else {
            exe_cmd.addArgs(&.{ "-I", "/usr/include/opencv4", "-I", "/usr/include/opencv5", "-I", "/usr/local/include", "-I", "/usr/include" });
        }
        for (header_files) |hdr| {
            exe_cmd.addFileInput(b.path(hdr));
        }
        for (cli_sources) |src| {
            exe_cmd.addFileArg(b.path(src));
        }
        for (opencv_libs) |lib_name| {
            exe_cmd.addArg(b.fmt("-l{s}", .{lib_name}));
        }
        exe_cmd.addArg("-o");
        const exe_out = exe_cmd.addOutputFileArg("vidicant_cli");
        const install_exe = b.addInstallFile(exe_out, "bin/vidicant_cli");
        b.getInstallStep().dependOn(&install_exe.step);
        return;
    }

    // macOS / Windows: Use Zig's native Clang toolchain + libc++
    const lib_mod = b.createModule(.{
        .target = target,
        .optimize = optimize,
    });
    lib_mod.addCSourceFiles(.{
        .files = &lib_sources,
        .flags = &.{ "-std=c++17", "-Wall", "-Wextra" },
    });
    lib_mod.addIncludePath(.{ .cwd_relative = "include" });
    configureOpenCV(b, lib_mod, target, opencv_path);
    for (opencv_libs) |lib_name| {
        lib_mod.linkSystemLibrary(lib_name, .{});
    }
    lib_mod.link_libc = true;
    lib_mod.link_libcpp = true;

    const lib = b.addLibrary(.{
        .name = "vidicant",
        .linkage = .dynamic,
        .root_module = lib_mod,
    });
    b.installArtifact(lib);

    if (install_to_pkg) {
        const copy_pkg = b.addSystemCommand(&.{"cp"});
        copy_pkg.addFileArg(lib.getEmittedBin());
        copy_pkg.addArg("vidicant/");
        b.getInstallStep().dependOn(&copy_pkg.step);
    }

    const exe_mod = b.createModule(.{
        .target = target,
        .optimize = optimize,
    });
    exe_mod.addCSourceFiles(.{
        .files = &cli_sources,
        .flags = &.{ "-std=c++17", "-Wall", "-Wextra" },
    });
    exe_mod.addIncludePath(.{ .cwd_relative = "include" });
    configureOpenCV(b, exe_mod, target, opencv_path);
    for (opencv_libs) |lib_name| {
        exe_mod.linkSystemLibrary(lib_name, .{});
    }
    exe_mod.link_libc = true;
    exe_mod.link_libcpp = true;

    const exe = b.addExecutable(.{
        .name = "vidicant_cli",
        .root_module = exe_mod,
    });
    b.installArtifact(exe);
}

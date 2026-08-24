const std = @import("std");

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
    const os = target.result.os.tag;

    // Linux: System OpenCV packages use GNU libstdc++ ABI.
    // Use host C++ compiler to guarantee ABI compatibility with system OpenCV.
    if (os == .linux) {
        const lib_cmd = b.addSystemCommand(&.{
            "c++",
            "-std=c++17",
            "-O3",
            "-fPIC",
            "-shared",
            "-I",
            "include",
            "-I",
            "/usr/include/opencv4",
            "-I",
            "/usr/include/opencv5",
            "-I",
            "/usr/local/include",
            "-I",
            "/usr/include",
            "src/image.cpp",
            "src/video.cpp",
            "src/controller.cpp",
            "src/vidicant_c_api.cpp",
            "-lopencv_core",
            "-lopencv_imgproc",
            "-lopencv_imgcodecs",
            "-lopencv_video",
            "-lopencv_videoio",
            "-lopencv_objdetect",
            "-lopencv_dnn",
            "-o",
        });
        const lib_out = lib_cmd.addOutputFileArg("libvidicant.so");
        const install_lib = b.addInstallFile(lib_out, "lib/libvidicant.so");
        b.getInstallStep().dependOn(&install_lib.step);

        const exe_cmd = b.addSystemCommand(&.{
            "c++",
            "-std=c++17",
            "-O3",
            "-I",
            "include",
            "-I",
            "/usr/include/opencv4",
            "-I",
            "/usr/include/opencv5",
            "-I",
            "/usr/local/include",
            "-I",
            "/usr/include",
            "src/main.cpp",
            "src/controller.cpp",
            "src/image.cpp",
            "src/video.cpp",
            "-lopencv_core",
            "-lopencv_imgproc",
            "-lopencv_imgcodecs",
            "-lopencv_video",
            "-lopencv_videoio",
            "-lopencv_objdetect",
            "-lopencv_dnn",
            "-o",
        });
        const exe_out = exe_cmd.addOutputFileArg("vidicant_cli");
        const install_exe = b.addInstallFile(exe_out, "bin/vidicant_cli");
        b.getInstallStep().dependOn(&install_exe.step);
        return;
    }

    // macOS / Windows: Use Zig's native Clang toolchain + libc++
    // 1. Module and Dynamic Library for Python / C-ABI (libvidicant.dylib / .dll)
    const lib_mod = b.createModule(.{
        .target = target,
        .optimize = optimize,
    });
    lib_mod.addCSourceFiles(.{
        .files = &.{
            "src/image.cpp",
            "src/video.cpp",
            "src/controller.cpp",
            "src/vidicant_c_api.cpp",
        },
        .flags = &.{
            "-std=c++17",
            "-Wall",
            "-Wextra",
        },
    });
    lib_mod.addIncludePath(.{ .cwd_relative = "include" });
    configureOpenCV(b, lib_mod, target, opencv_path);

    lib_mod.linkSystemLibrary("opencv_core", .{});
    lib_mod.linkSystemLibrary("opencv_imgproc", .{});
    lib_mod.linkSystemLibrary("opencv_imgcodecs", .{});
    lib_mod.linkSystemLibrary("opencv_video", .{});
    lib_mod.linkSystemLibrary("opencv_videoio", .{});
    lib_mod.linkSystemLibrary("opencv_objdetect", .{});
    lib_mod.linkSystemLibrary("opencv_dnn", .{});
    lib_mod.link_libc = true;
    lib_mod.link_libcpp = true;

    const lib = b.addLibrary(.{
        .name = "vidicant",
        .linkage = .dynamic,
        .root_module = lib_mod,
    });

    b.installArtifact(lib);

    // 2. Module and Executable for CLI (vidicant_cli)
    const exe_mod = b.createModule(.{
        .target = target,
        .optimize = optimize,
    });
    exe_mod.addCSourceFiles(.{
        .files = &.{
            "src/main.cpp",
            "src/controller.cpp",
            "src/image.cpp",
            "src/video.cpp",
        },
        .flags = &.{
            "-std=c++17",
            "-Wall",
            "-Wextra",
        },
    });
    exe_mod.addIncludePath(.{ .cwd_relative = "include" });
    configureOpenCV(b, exe_mod, target, opencv_path);

    exe_mod.linkSystemLibrary("opencv_core", .{});
    exe_mod.linkSystemLibrary("opencv_imgproc", .{});
    exe_mod.linkSystemLibrary("opencv_imgcodecs", .{});
    exe_mod.linkSystemLibrary("opencv_video", .{});
    exe_mod.linkSystemLibrary("opencv_videoio", .{});
    exe_mod.linkSystemLibrary("opencv_objdetect", .{});
    exe_mod.linkSystemLibrary("opencv_dnn", .{});
    exe_mod.link_libc = true;
    exe_mod.link_libcpp = true;

    const exe = b.addExecutable(.{
        .name = "vidicant_cli",
        .root_module = exe_mod,
    });

    b.installArtifact(exe);
}

include(FetchContent)

function (get_google_benchmark)
    FetchContent_Declare(
        googlebenchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG        v1.5.0
    )
    FetchContent_MakeAvailable(googlebenchmark)
endfunction()

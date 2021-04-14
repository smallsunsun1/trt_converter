include(FetchContent)

function (get_google_test)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        release-1.8.0
    )
    FetchContent_MakeAvailable(googletest)
endfunction()

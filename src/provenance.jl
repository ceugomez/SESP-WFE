# provenance.jl
# cgf cego6160@colorado.edu 11.4.26
#
# Provenance attributes stamped on every result file, so a number can be traced to the code
# that made it. `git_dirty = true` means the working tree had uncommitted changes when the
# file was written — the commit alone does not then reproduce it.
using Dates

function _git(args...)
    root = normpath(joinpath(@__DIR__, ".."))
    try
        return strip(read(Cmd(`git -C $root $args`), String))
    catch
        return "unknown"
    end
end

"Dict of provenance attributes for a result file written by `script` (pass @__FILE__)."
function provenance_attrs(script::AbstractString)
    status = _git("status", "--porcelain")
    return Dict{String,Any}(
        "script"     => relpath(script, normpath(joinpath(@__DIR__, ".."))),
        "git_commit" => _git("rev-parse", "HEAD"),
        "git_dirty"  => status == "unknown" ? "unknown" : string(!isempty(status)),
        "created"    => string(now(UTC)) * "Z",
        "host"       => gethostname(),
        "julia"      => string(VERSION),
    )
end

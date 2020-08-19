import MLJBase # TODO: remove the dependency on MLJBase.jl

#### BEGIN code to make predict_joint work on machines. Remove this once it is merged in MMI upstream.

const OPERATIONS = (:predict_joint,)

# for operation in OPERATIONS
#
#     if operation != :inverse_transform
#
#         ex = quote
#             # 0. operations on machs, given empty data:
#             function $(operation)(mach::MLJBase.Machine; rows=:)
#                 # Base.depwarn("`$($operation)(mach)` and "*
#                 #              "`$($operation)(mach, rows=...)` are "*
#                 #              "deprecated. Data or nodes "*
#                 #              "should be explictly specified, "*
#                 #              "as in `$($operation)(mach, X)`. ",
#                 #              Base.Core.Typeof($operation).name.mt.name)
#                 if isempty(mach.args) # deserialized machine with no data
#                     throw(ArgumentError("Calling $($operation) on a "*
#                                         "deserialized machine with no data "*
#                                         "bound to it. "))
#                 end
#                 return ($operation)(mach, mach.args[1](rows=rows))
#             end
#         end
#         eval(ex)
#
#     end
# end

# _symbol(f) = Base.Core.Typeof(f).name.mt.name

for operation in OPERATIONS

    ex = quote
        # 1. operations on machines, given *concrete* data:
        function $operation(mach::MLJBase.Machine, Xraw)
            if mach.state > 0
                return $(operation)(mach.model, mach.fitresult,
                                    Xraw)
            else
                error("$mach has not been trained.")
            end
        end

        # function $operation(mach::MLJBase.Machine{<:MMI.Static}, Xraw, Xraw_more...)
        #     isdefined(mach, :fitresult) || (mach.fitresult = nothing)
        #     return $(operation)(mach.model, mach.fitresult,
        #                             Xraw, Xraw_more...)
        # end

        # 2. operations on machines, given *dynamic* data (nodes):
        # $operation(mach::MLJBase.Machine, X::MLJBase.AbstractNode) =
        #     node($(operation), mach, X)

        # $operation(mach::MLJBase.Machine{<:MMI.Static}, X::MLJBase.AbstractNode, Xmore::MLJBase.AbstractNode...) =
        #     node($(operation), mach, X, Xmore...)
    end
    eval(ex)
end

## SURROGATE AND COMPOSITE MODELS

# for operation in [:predict_joint,]
#     ex = quote
#         $operation(model::Union{MLJBase.Composite,MLJBase.Surrogate}, fitresult,X) =
#             fitresult.$operation(X)
#     end
#     eval(ex)
# end

#### END code to make predict_joint work on machines. Remove this once it is merged in MMI upstream.

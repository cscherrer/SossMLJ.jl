import MLJBase

const OPERATIONS = (:predict_particles,)

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
    end
    eval(ex)
end

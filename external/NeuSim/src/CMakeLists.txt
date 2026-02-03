aux_source_directory(. SOURCES)
aux_source_directory(noc SOURCES)
aux_source_directory(noc/topo SOURCES)

message(${SOURCES})

add_library(neu_sim ${SOURCES})
target_include_directories(neu_sim PUBLIC ${NC_SIM_INCLUDE_DIR})
set_target_properties(neu_sim PROPERTIES POSITION_INDEPENDENT_CODE ON)

module flip(
    input logic  clk, rst, 
    output logic flipper
);    
    always_ff @(posedge clk) begin
        flipper <= ~flipper;
    end

endmodule